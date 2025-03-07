#!/usr/bin/env python
"""
This class will generate a time series plot of PSWS Ground Magnetometer Data.
"""
import os
import shutil
import datetime
import tqdm
import logging
logger  = logging.getLogger(__name__)

import pickle

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy import signal

import matplotlib as mpl
from matplotlib import pyplot as plt

#import solarContext

mpl.rcParams['font.size']      = 12
mpl.rcParams['font.weight']    = 'bold'
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.xmargin']   = 0

prmds   = {}
prmds['rx'] = prmd = {}
prmd['label'] = 'x'
prmds['ry'] = prmd = {}
prmd['label'] = 'y'
prmds['rz'] = prmd = {}
prmd['label'] = 'z'
prmds['rH'] = prmd = {}
prmd['label'] = 'H'

def gmag_fname(station,datetime,suffix='.log'):
    """
    Return the filename for a HamSCI PSWS ground magnetometer data file.
    """
    date_str    = datetime.strftime('%Y%m%d')
    fname       = '{!s}-{!s}-runmag{!s}'.format(station,date_str,suffix)
    return fname

def list_dates(sDatetime,eDatetime):
    """
    Create a list of dates between sDatetime and eDatetime.
    """

    sDate   = datetime.datetime(sDatetime.year,sDatetime.month,sDatetime.day)
    date_list   = [sDate]
    while date_list[-1] < eDatetime:
        next_date = date_list[-1] + datetime.timedelta(days=1)
        if next_date == eDatetime: 
            break
        date_list.append(next_date)

    return date_list

def find_gmag_files(station,sDatetime,eDatetime,data_dir='.',suffix='.log'):
    """
    Find HamSCI PSWS Ground Magnetometer data files for a particular
    station and date range.
    """

    dates   = list_dates(sDatetime,eDatetime)
    fpaths  = []
    for date in dates:
        fname = gmag_fname(station,date,suffix=suffix)
        fpath = os.path.join(data_dir,fname)
        if not os.path.exists(fpath):
            logger.warning('FILE NOT FOUND: {!s}'.format(fpath)) 
            continue
        fpaths.append(fpath)

    return fpaths

class PSWS_GMAG(object):
    def __init__(self,station,sDatetime,eDatetime,
            lat=None,lon=None,
            data_dir=os.path.join('data','psws_gmag'),suffix='.log.bz2',
            **kwargs):

        gmag_files  = find_gmag_files(station,sDatetime,eDatetime,
                            data_dir=data_dir,suffix=suffix)

        meta = {}
        meta['label']       = '{!s} Ground Magnetometer'.format(station.upper())
        meta['station']     = station
        meta['sDatetime']   = sDatetime
        meta['eDatetime']   = eDatetime
        meta['data_dir']    = data_dir
        meta['files']       = gmag_files
        meta['lat']         = lat
        meta['lon']         = lon
        meta['solar_lat']   = lat # Latitude where all solar calcultions are made
        meta['solar_lon']   = lon # Longitude where all solar calcultions are made
        self.meta           = meta

        self.load_data()

        self.process_data(**kwargs)
        self.rolling_mean(**kwargs)


    def load_data(self):
        files = self.meta.get('files')

        df  = []
        for fpath in tqdm.tqdm(files,desc='Loading PSWS GMAG Data',dynamic_ncols=True):
            pkl_path = fpath + '.p'

            if os.path.exists(pkl_path):
                msg = 'Loading: {!s}'.format(pkl_path)
                logging.info(msg)
                tqdm.tqdm.write(msg)
                with open(pkl_path,'rb') as fl:
                    dft = pickle.load(fl)
            else:
                msg = 'Loading: {!s}'.format(fpath)
                logging.info(msg)
                tqdm.tqdm.write(msg)
                dft  = pd.read_json(fpath,lines=True,convert_dates=['ts'])
                with open(pkl_path,'wb') as fl:
                    pickle.dump(dft,fl)
            df.append(dft)

        df  = pd.concat(df,ignore_index=True)
        df  = df.rename(columns={'ts':'UTC'})
        tf  = np.logical_and(df['UTC'] >= self.meta['sDatetime'], df['UTC'] < self.meta['eDatetime'])
        df_raw  = df[tf]

        # Apply Hyomin's Recommended Coordinate Correction for W2NAF
#        Bx_new = Bz
#        By_new = By
#        Bz_new = -Bx
#
#        With this conversion, I see NO difference in your data (W2NAF) in comparison to the nearby USGS mag (Fredericksburg). 
        rx_new  = df_raw['rz']
        ry_new  = df_raw['ry']
        rz_new  = -df_raw['rx']

        df_raw['rx'] = rx_new
        df_raw['ry'] = ry_new
        df_raw['rz'] = rz_new

        # Compute remote Horiztonal magnitude
        df_raw['rH'] = np.sqrt( df_raw['rx']**2 + df_raw['ry']**2 )

        # Store data into dictionaries.
        data = {}
        data['raw']          = {}
        data['raw']['df']    = df_raw
        data['raw']['label'] = 'Raw Data'
        self.data = data
        
        self.df_raw   = df

    def plot_figure(self,png_fpath='output.png',figsize=(16,5),**kwargs):

        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1,1,1)

        result  = self.plot_ax(ax,**kwargs)

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        plt.close(fig)
    
    def plot_ax(self,ax,data_sets=['raw'],
            xlim=None,ylim=None,
            prms = ['rx','ry','rz'],data_set_kws=None,
            solar_lat=None,solar_lon=None,
            overlaySolarElevation=True,overlayEclipse=False):

        station = self.meta.get('station')
        fig     = ax.get_figure()

        if len(data_sets) == 2:
            ds_kws  = []
            ds_kws.append({'lw':2})
            ds_kws.append({'lw':1,'alpha':0.5})
        else:
            ds_kws  = [{}] * len(data_sets)

        for ds_inx,data_set in enumerate(data_sets):
            ds          = self.data.get(data_set)
            ds_kw       = ds_kws[ds_inx]
            ds_label    = ds.get('label')
            df          = ds.get('df')

            ax.set_prop_cycle(None) # Reset color cycle
            for prm in prms:

                prmd    = prmds.get(prm,{})
                xx      = df['UTC']

                vals    = df[prm]
                mu_vals = np.nanmean(vals)
                yy      = vals - mu_vals

                if ds_inx == 0:
                    lbl     = prmd.get('label',prm)
                    lbl     = r'{!s} ($\mu$ = {:.0f} nT)'.format(lbl,mu_vals)
                else:
                    lbl     = None

                if prm == 'rH':
                    rH_kw = ds_kw.copy()
                    rH_kw['lw'] = 5
                    ax.plot(xx,yy,label=lbl,**rH_kw)
                else:
                    if data_set_kws is not None:
                        ds_kw.update(data_set_kws[ds_inx])
                    ax.plot(xx,yy,label=lbl,**ds_kw)

            if ds_inx == 0:
                title   = '{!s}: {!s} Ground Magnetometer'.format(ds_label, station.upper())
                ax.set_title(title)


        ax.legend(loc='lower left',ncols=4)

        ylabel  = 'nT'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('UTC Date')

#        ##### Overlay Solar Eclipse and Elevation
#        _xlim   = mpl.dates.num2date(ax.get_xlim())
#        sTime   = _xlim[0]
#        eTime   = _xlim[1]
#
#        solar_lat   = self.meta.get('solar_lat',solar_lat)
#        solar_lon   = self.meta.get('solar_lon',solar_lon)
#
#        sts = solarContext.solarTimeseries(sTime,eTime,solar_lat,solar_lon)
#        if overlaySolarElevation:
#            sts.overlaySolarElevation(ax)
#
#        if overlayEclipse:
#            sts.overlayEclipse(ax)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        result  = {}
        result['title'] = title
        return result

    def process_data(self,profile='standard',resample_rate=datetime.timedelta(minutes=1),**kwargs):
        tic_0 = datetime.datetime.now()
        print('Processing data using "{!s}" profile...'.format(profile))
        print('')
        if profile == 'standard':
            data_set_in = 'raw'
            data_set    = 'resampled'

            print('Resampling data with {!s} minute cadence...'.format(resample_rate.total_seconds()/60.))
            tic = datetime.datetime.now()
            self.resample_data(resample_rate=resample_rate,method='mean',
                              data_set_in=data_set_in,data_set_out=data_set)
            toc = datetime.datetime.now()
            print('  Resampling Time: {!s}'.format(toc-tic))
            print()

#            print('Computing Solar Local Time on resampled...')
#            tic = datetime.datetime.now()
#            self.calculate_solar_time(data_set)
#            toc = datetime.datetime.now()
#            print('  Solar Time Computation Time: {!s}'.format(toc-tic))
#            print()

        toc_0 = datetime.datetime.now()
        print('')
        print('Total Processing Time: {!s}'.format(toc_0-tic_0))

    def rolling_mean(self,rolling_window=datetime.timedelta(minutes=3),**kwargs):
        tic_0 = datetime.datetime.now()
        print('Applying rolling...')
        print('')
        data_set_in = 'resampled'
        data_set_out= 'rolling'

        df   = self.data[data_set_in]['df'].copy()
        rs_df = df.rolling(rolling_window,on='UTC',center=True).mean()

        tmp          = {}
        tmp['df']    = rs_df
        tmp['label'] = 'Rolling mean ({:.1f} min)'.format(rolling_window.total_seconds()/60.)
        self.data[data_set_out] = tmp

        toc_0 = datetime.datetime.now()
        print('')
        print('Total Processing Time: {!s}'.format(toc_0-tic_0))
    
    def resample_data(self,resample_rate,on='UTC',method='mean',
                          data_set_in='raw',data_set_out='resampled'):
        
        df   = self.data[data_set_in]['df'].copy()

        if len(df) == 0:
            rs_df = df.copy()
        else:
            # Create the list of datetimes that we want to resample to.
            # Find the start and end times of the array.
            sTime = df['UTC'].min()
            eTime = df['UTC'].max()

            tzinfo= sTime.tzinfo

            # Break
            sYr  = sTime.year
            sMon = sTime.month
            sDy  = sTime.day
            sHr  = sTime.hour
            sMin = sTime.minute
            sSec = sTime.second
            resample_sTime = datetime.datetime(sYr,sMon,sDy,sHr,sMin,sSec,tzinfo=tzinfo)

            eYr  = eTime.year
            eMon = eTime.month
            eDy  = eTime.day
            eHr  = eTime.hour
            eMin = eTime.minute
            eSec = eTime.second
            resample_eTime = datetime.datetime(eYr,eMon,eDy,eHr,eMin,eSec,tzinfo=tzinfo)

            # Remove LMT column if it exists because it cannot be resampled.
            if 'LMT' in df.keys():
                df = df.drop('LMT',axis=1)

            cols        = df.keys()
            df          = df.set_index(on) # Need to make UTC column index for interpolation to work.
            df          = df.drop_duplicates()

            df          = df[~df.index.duplicated(keep='first')] # Make sure there are no duplicated indices.

            rs_df       = df.resample(resample_rate,origin=resample_sTime)
            if method == 'mean': 
                rs_df = rs_df.mean()
            else:
                rs_df = rs_df.interpolate(method='linear')

            rs_df       = rs_df.copy()
            rs_df[on]   = rs_df.index
            rs_df.index = np.arange(len(rs_df)) 

            # Put Columns back in original order.
            rs_df       = rs_df[cols].copy()
        
        tmp          = {}
        tmp['df']    = rs_df
        tmp['label'] = 'Resampled Data (dt = {!s} s)'.format(resample_rate.total_seconds())
        tmp['Ts']    = resample_rate.total_seconds()
        self.data[data_set_out] = tmp

if __name__ == '__main__':
    output_dir = os.path.join('output','psws_gmag')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rd = {}
    rd['station']   = 'w2naf'
    rd['lat']       =  41.335116
    rd['lon']       = -75.600692
    rd['sDatetime'] = datetime.datetime(2024,4,8)
    rd['eDatetime'] = datetime.datetime(2024,4,9)
    rd['resample_rate']  = datetime.timedelta(minutes=1)
    rd['rolling_window'] = datetime.timedelta(minutes=5)

    plot_dict = {}
    plot_dict['overlayEclipse'] = True
#    plot_dict['xlim'] = (datetime.datetime(2024,4,8,12), datetime.datetime(2024,4,9))
    plot_dict['ylim'] = (-300,300)
#    plot_dict['prms'] = ['rH']
    plot_dict['prms'] = ['rH','rx','ry']

    gmag        = PSWS_GMAG(**rd)

    sDt_str     = rd['sDatetime'].strftime('%Y%m%d.%H%M')
    eDt_str     = rd['eDatetime'].strftime('%Y%m%d.%H%M')
#    data_sets   = ['resampled','raw']
    data_sets   = ['rolling','resampled']

#    for data_set in data_sets:
#        png_fname   = '{!s}-{!s}-{!s}_{!s}_gmag.png'.format(rd['station'],sDt_str,eDt_str,data_set)
#        png_fpath   = os.path.join(output_dir,png_fname)
#        gmag.plot_figure(data_sets=[data_set],png_fpath=png_fpath,**plot_dict)
#        print(png_fpath)

    png_fname   = '{!s}-{!s}-{!s}_gmag.png'.format(rd['station'],sDt_str,eDt_str)
    png_fpath   = os.path.join(output_dir,png_fname)
    gmag.plot_figure(data_sets=data_sets,png_fpath=png_fpath,**plot_dict)
    print(png_fpath)
