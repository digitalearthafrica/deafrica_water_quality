#from _WQ_functions import dummy_function
# importing required module
import os
import numpy  as np
import xarray as xr
import matplotlib.pyplot as plt
import gc   # garbage collection
import pandas as pd
import sys
import datacube


# appending a path
sys.path.append('/home/jovyan/dev/deafrica_water_quality/WP1.2')
from _WQ_functions import trophic_state
from _WQ_functions import instruments_list  
from _WQ_functions import *  
#from WQ_functions import NDVI
#from _WQ_functions import FAI
#from WQ_functions import NDVI



# --- this cell is functions ----
# --- calculate summary statistics 
#for now I have kept these functions in a notebook for the markdown

def wp22_dummy(s='nonesense'):
    print('function wp22_dummy is now running -->'+s)
    trophic_state()
    #instruments_list()
    return()

# --------------------------------------------------------------------------------
# --- a function to calculate the water_pixel_count (over water areas), 
# the coveragepercent (over water areas), and 
# the qapasspercent (of available data over water areas) for each observation 
def determine_water_area (ds,ds_annual):
    year      = pd.DatetimeIndex(ds['time']).year[int(ds.time.size/2)]  # the year at the centre of the monitoring interval
    water_5yr = ds_annual.water_5yr.sel(time=str(year)).squeeze()       # a data array with xy shape
    water_pixel_count = water_5yr.count()

    # --- calculate the proportion of observations that are present and valid, at each time-step
    ds['data_coverage']   =  (ds.where(~np.isnan(ds.qa_score)).where(water_5yr == 1).qa_score.count(dim=('x','y'))/water_pixel_count)

    # --- calculate the proportion of valid observations that pass QA
    ds['data_qapass']     =  (ds.where(ds.qa_score >0)        .where(water_5yr == 1).qa_score.count(dim=('x','y'))/water_pixel_count)

    # --- save the water pixel count as a variable in the dataset ...
    ds['water_pixel_count'] = water_pixel_count
    return(ds)
    
# --------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# --- a function to set the geomedian years, based on the timeframe given. At least 5 years are needed. 
def geomedian_instruments_and_years(params):
    instruments_to_use = {
        'oli_agm'  : {'use': True },
        'msi_agm'  : {'use': True },
        'tm_agm'   : {'use': True },
        'wofs_ann' : {'use': True },
        'wofs_all' : {'use': False},
        'oli'      : {'use': False},
        'msi'      : {'use': False },
        'tm'       : {'use': False },
        'tirs'       : {'use': False },
        'wofs_ann' : {'use': True },
        'wofs_all' : {'use': False},
        }
    # --- although the analysis window might be only a year or two, we will use at least 5 years of geomedian data to 
    #     identify areas of water, and we will run a geomedian analysis over this timeframe. 
    
    gm_year_start = np.min([params['year1'],params['year2']-5])
    gm_year_end   = np.max([params['year2'],params['year1']+5])
    gm_year_end   = np.min([gm_year_end,2025])
    gm_year_start = np.min([gm_year_start,gm_year_end-5])
    gm_year_start = np.max([gm_year_start,2000])
    gm_years = [int(gm_year_start),int(gm_year_end)]
 
    return(gm_years,instruments_to_use)

# ------------------------------------------------------------------------------------------
# a function to calcultate the NDVI; for non-geomedian datasets at this point since its already covered for geomedians
# This function calculates the NDVI for a designated instrument.
# this is a copy since I can't get WP22_functions.py to recognise it in WQ_functions.py 
# (in contrast, recognising FAI function is not a problem...)

def NDVI(ds, instrument, test=False):
    instr = instrument
    ndvi_bands = {}
    ndvi_bands['tm']  = ['tm04','tm03']
    ndvi_bands['oli'] = ['oli05','oli04']
    ndvi_bands['msi'] = ['msi8a','msi04']
    if not instr in ndvi_bands.keys():
        print('! -- invalid instrument, NDVI will be calculated as zero --- !')
        return(0)
    return(
        (ds[ndvi_bands[instr][0]] - ds[ndvi_bands[instr][1]]) / \
        (ds[ndvi_bands[instr][0]] + ds[ndvi_bands[instr][1]]) 
        )



# ---------------------------------------------------------------------------------------------
def calc_ndvi(ds,watermask,threshold = 0.0):
    ds['ndvi'] = ('time','y','x'),np.zeros([ds.qa_score.sizes['time'],ds.qa_score.sizes['y'],ds.qa_score.sizes['x']])*np.nan
    if 'tm'  in ds.data_vars:    
        ds['ndvi'].loc[ds.tm ==True] = NDVI(ds.where(ds.ndvi.loc[ds.tm== True]),'tm' )
    if 'oli' in ds.data_vars:    
        ds['ndvi'].loc[ds.oli==True] = NDVI(ds.where(ds.ndvi.loc[ds.oli==True]),'oli')
    if 'msi' in ds.data_vars:    
        ds['ndvi'].loc[ds.msi==True] = NDVI(ds.where(ds.ndvi.loc[ds.msi==True]),'msi')

    ds['ndvi'] = ds.ndvi.where(ds.ndvi > threshold).where(watermask==1).where(ds.qa_score >0) 
    return(ds)

# ---------------------------------------------------------------------------------------------
def calc_fai(ds,watermask,threshold = 0.0):
    ds['fai'] = ('time','y','x'),np.zeros([ds.qa_score.sizes['time'],ds.qa_score.sizes['y'],ds.qa_score.sizes['x']])*np.nan
    if 'tm'  in ds.data_vars:    
        ds['fai'].loc[ds.tm ==True] = FAI(ds.where(ds.fai.loc[ds.tm== True]),'tm' )
    if 'oli' in ds.data_vars:    
        ds['fai'].loc[ds.oli==True] = FAI(ds.where(ds.fai.loc[ds.oli==True]),'oli')
    if 'msi' in ds.data_vars:    
        ds['fai'].loc[ds.msi==True] = FAI(ds.where(ds.fai.loc[ds.msi==True]),'msi')

    ds['fai'] = ds.fai.where(ds.fai > threshold).where(watermask==1).where(ds.qa_score >0) 
    return(ds)
    
# ---------------------------------------------------------------------------------------------
# --- a function that adds boolean variables that retain the source instrument, and a qa_score variable. 
# Stil Testing that this behaves correctly when the qa_score is pre-set using the agm tests ....
def add_vars_and_combine_datasets(data_list):
    # --- setting qa_score to nan, 0 or 1
    for instrument in data_list.keys():
        # --- select the relevant dataset ---
        ds        = data_list[instrument]
        ds[instrument] = ('time'), np.ones(ds.time.sizes['time']).astype('bool')                    # --- a boolean variable with a value of 'True' ---
        if not 'qa_score' in ds.data_vars:
            ds['qa_score'] = ('time','y','x'), np.zeros([ds.sizes['time'],ds.sizes['y'],ds.sizes['x']]) # --- a "qa_score" variable and set to zero ----

        # --- attempt to set the value of the qa_score, based on the pixel quality data provided with the data.  
        if instrument == 'tm':  
            ds['qa_score'] = xr.where(ds.tm,
                                      xr.where(np.isnan(ds.tm_qa),np.nan,ds.qa_score),ds.qa_score)   # --- sets the qa_score to nan where there is nodata (should not happen)
            # --- frankly, this is pretty flaky; I don't trust that the data are consistenly labeled and with TM I am picking specific values
            # ---- values of 5504 should be correct, but 5503 and 5505 also appear. 
        
            ds['qa_score'] = xr.where(np.isin(ds.tm_qa,[5503,5504,5505,5440]),ds.qa_score,0) #DEA mess around with the pixel values so that they sort of conform to OPI, but not really.
            ds['qa_score'] = xr.where(ds.tm_qa==1 , np.nan , ds.qa_score) #values of 1 are 'fill', set to nan

        if instrument == 'oli': 
            ds['qa_score'] = xr.where(ds.oli,
                                      xr.where(np.isnan(ds.oli_qa),np.nan,ds.qa_score),ds.qa_score)  # --- sets the qa_score to nan where there is no coverage
            ds['qa_score'] = xr.where(ds.oli_qa<=21952 , ds.qa_score,0) 
            ds['qa_score'] = xr.where(ds.oli_qa==1 , np.nan , ds.qa_score) #values of 1 are 'fill', set to nan
            # oli qa scores are found here : https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
            # 21952 corresponds to a clear pixel over water. This is the base for use here.

        if instrument == 'msi': 
            ds['qa_score'] = xr.where(ds.msi,
                                      xr.where(np.isnan(ds.msi_qa),np.nan,ds.qa_score),ds.qa_score)   # --- sets the qa_score to nan where there is no coverage (should not happen)
            ds['qa_score'] = xr.where(ds.msi_qa < 7,ds.qa_score,0)
            ds['qa_score'] = xr.where(ds.msi_qa==1 , np.nan , ds.qa_score) #values of 1 are 'fill', set to nan

            
    # --- now combine all the instrument datasets to a single dataset ---
    first = True
    for name in data_list.keys():
        if first:
            full_ds    = data_list[name]
            first = False
        else:
            full_ds = full_ds.combine_first(data_list[name])
    return(full_ds)


# --------------------------------------------------------------------------------------
# ---- This is a function to pull out the data for each geomedian, and build it into a single multi-instrument geomedian dataset
# this version is from the FAI notebook; it is more mature than the original which did not pass all the arguments in.

def build_agm_dataset(parameters,instruments_to_use,verbose=True):
    # --- loads the 'data products' from the data cube collections
    # --- returns a single dataset of uniform spatial resolution
    if verbose : print('\nBuilding the Geomedian dataset:')

    spacetime_domain = parameters['xyt'].copy()
    spacetime_domain['time']  = (str(parameters['year1']),str(parameters['year2']))   # --- use the gemedan year range
    
    products = { 'tm_agm' :["gm_ls5_ls7_annual"],
                'oli_agm' :["gm_ls8_annual","gm_ls8_ls9_annual"],
                'msi_agm' :["gm_s2_annual"],
                #'tirs'    :["ls5_st","ls7_st","ls8_st","ls9_st"],
                'wofs_ann':["wofs_ls_summary_annual"],
                'wofs_all':["wofs_ls_summary_alltime"],
               }

    instruments,measurements,rename_dict = instruments_list(instruments_to_use) 
    datasets = {}
    dc = datacube.Datacube(app='build_agm_dataset')
    for instrument in list(instruments_to_use.keys()):
        if instruments_to_use[instrument]['use'] :
            if verbose : print('loading data for ',instrument,'...')
            datasets[instrument] = dc.load(product=(products[instrument]),
                                 **spacetime_domain,
                                 **{'measurements': measurements[instrument]},
                                 output_crs='epsg:6933',
                                 resolution=parameters['grid_resolution'],
                                 align=(0,0),
                                 resampling=parameters['resampling_option'],)
    
    #added a CRS since temperature data crashes without it

    #separating the rename step out:
    #rename the measurements to standardised variable names,

    for instrument in list(instruments_to_use.keys()):
        if instruments_to_use[instrument]['use']:      
            datasets[instrument] = rename_vars_robust(datasets[instrument],rename_dict[instrument],False)       
     
    # .... and build a list of datasets to merge:
    mergelist = []; i = 0
    first = True
    for instrument in list(instruments_to_use.keys()):
        if instruments_to_use[instrument]['use'] and not instrument == 'tirs':      
            #datasets[instrument] = rename_vars_robust(datasets[instrument],rename_dict[instrument],False)       
            if first :
                first = False
                dataset = datasets[instrument]
            else:
                dataset = dataset.combine_first(datasets[instrument])
            mergelist.append(datasets[instrument])
    return(dataset)



# ---------------------------------------------------------------------------------------------
def build_dataset (spacetime_domain,
                   instruments_to_use, 
                   products, 
                   measurements, 
                   grid_resolution,
                   resampling_option,
                   verbose=True) :
    if instruments_to_use['oli']['use']:
        if verbose : print('building the oli dataset...')
        instrument = 'oli'
        # --- load oli data
        # Load available data from all three Landsat satellites
        dc = datacube.Datacube(app='building_dataset')
        ds_oli = dc.load(product=(products[instrument]),
                                 **spacetime_domain,
                                 **{'measurements': measurements[instrument]},
                                 output_crs='epsg:6933',
                                 group_by ='solar_day',
                                 resolution=grid_resolution,
                                 align=(0,0),
                                 resampling=resampling_option,)
        # --- re-name the variables for the sake of sanity --- 
        ds_oli = rename_vars_robust(ds_oli,rename_dict['oli'])
        # --- set zeros to nans and re-scale 
        for var in ds_oli.data_vars:
            ds_oli[var] = xr.where(ds_oli[var]>0,ds_oli[var],np.nan)
            if not var == 'oli_pq':
                ds_oli[var] = ((ds_oli[var] * 0.0000275) - 0.2) * 10000

        data_list['oli'] = ds_oli
        if verbose : print('... done.')
    

    if instruments_to_use['msi']['use']:
        if verbose : print('building the msi dataset....')
        instrument = 'msi'
        test = True
        # --- load msi data
        ds_msi = dc.load(product=(products[instrument]),
                                 **spacetime_domain,
                                 **{'measurements': measurements[instrument]},
                                 output_crs='epsg:6933',
                                 group_by ='solar_day',
                                 resolution=grid_resolution,
                                 align=(0,0),
                                 resampling=resampling_option,)
        # --- re-name the variables for the sake of sanity --- 
        ds_msi = rename_vars_robust(ds_msi,rename_dict['msi'])
        # --- set zeros to nans and re-scale 
        for var in ds_msi.data_vars:
            ds_msi[var] = xr.where(ds_msi[var]>0,ds_msi[var],np.nan)
            if not var == 'msi_pq':
             ds_msi[var] = ds_msi[var] #- 1000  # offset required for variables other than the pq ??
        
        data_list['msi'] = ds_msi
        if verbose : print('... done.')
    
    if instruments_to_use['tm']['use']:
        if verbose : print('building the tm dataset ...')
        instrument = 'tm'
        # --- load tm data
        ds_tm = dc.load(product=(products[instrument]),
                                 **spacetime_domain,
                                 **{'measurements': measurements[instrument]},
                                 output_crs='epsg:6933',
                                 group_by ='solar_day',
                                 resolution=grid_resolution,
                                 align=(0,0),
                                 resampling=resampling_option,)

        # --- re-name the variables for the sake of sanity --- 
        ds_tm = rename_vars_robust(ds_tm,rename_dict['tm'])
    

        # --- set zeros to nans and re-scale 
        for var in ds_tm.data_vars:
            ds_tm[var] = xr.where(ds_tm[var]>0,ds_tm[var],np.nan)
            if not var == 'tm_pq':
                ds_tm[var] = ((ds_tm[var] * 0.0000275) - 0.2) * 10000

        data_list['tm'] = ds_tm
        if verbose : print('... done.')

    #yet to switch this on...
    if instruments_to_use['tirs']['use'] and False:
        test = True
        if verbose : print('building the tirs dataset ...')
        instrument = 'tirs'
        data_list.append(instrument)
        # --- load tirs data
        ds_tm = dc.load(product=(products[instrument]),
                                 **spacetime_domain,
                                 **{'measurements': measurements[instrument]},
                                 output_crs='epsg:6933',
                                 group_by ='solar_day',
                                 resolution=grid_resolution,
                                 align=(0,0),
                                 resampling=resampling_option,)

        # --- re-name the variables for the sake of sanity --- 
        ds_tirs = rename_vars_robust(ds_tirs,rename_dict['tirs'])

        # --- set zeros to nans and re-scale 
        for var in ds_tirs.data_vars:
            ds_tm[var] = xr.where(ds_tm[var]>0,ds_tm[var],np.nan)
            if not var == 'tm_pq':
                ds_tm[var] = ((ds_tm[var] * 0.0000275) - 0.2) * 10000
        data_list['tirs']=ds_tirs
        if verbose : print('... done.')

    print('Data list = ',data_list.keys(),'\n')
    print('... instrument datasets complete\n')
    return(data_list)
    

# ---------------------------------------------------------------------------------------------
def calc_scale_and_offset(ds,verbose=False):
    # --- a function to calculate the scale and offset for level 0 normalisaton 
    # --- any variables to not include? I don't think this is needed ...
    exclude = ['chla_tebbs_oli','chla_tebbs_msi','chla_tebbs_tm']
    exclude = []
    refalgs = {}
    refalgs['chla'] = 'chla_modis2b_msi'
    refalgs['tss' ] = 'tsm_lym_oli'
    reftime    =  ds.time[0:24]   # the interval from which to gather the reference distributions
    targettime =  ds.time[0:24]   # the interval from which to gather the target distributions
    
    for var in ('tss','chla'):
        refalg = refalgs[var]
        ds[var+'_offset'] = (var+'_measure'), np.zeros((ds.sizes[var+'_measure']))
        ds[var+'_scale' ] = (var+'_measure'), np.zeros((ds.sizes[var+'_measure']))
        if var == 'tss':
            refmed = ds.loc[dict(time = (reftime), tss_measure = refalg, quantile=(ds['quantile'][50]))][var].median().values
            ref01  = ds.loc[dict(time = (reftime), tss_measure = refalg, quantile=(ds['quantile'][1] ))][var].median().values
            if verbose : print(refmed,ref01)
        if var == 'chla':
            refmed = ds.loc[dict(time = (reftime), chla_measure = refalg, quantile=(ds['quantile'][50]))][var].median().values
            ref01  = ds.loc[dict(time = (reftime), chla_measure = refalg, quantile=(ds['quantile'][1] ))][var].median().values
            if verbose : print(refmed,ref01)
      
        
        for name in (set(ds[var+'_measure'].values) - set(exclude)):
            if var == 'tss':
                med  = ds.loc[dict(time = (targettime), tss_measure = name, quantile=(ds['quantile'][50]))][var].median().values
                q_01 = ds.loc[dict(time = (targettime), tss_measure = name, quantile=(ds['quantile'][ 1]))][var].median().values
                scale  =  (refmed - ref01) / (med - q_01)
                offset =  refmed - med*scale        
                ds[var+'_offset'].loc[dict(tss_measure=name)] = offset
                ds[var+'_scale' ].loc[dict(tss_measure=name)] = scale
            if var == 'chla':
                med  = ds.loc[dict(time = (targettime), chla_measure = name, quantile=(ds['quantile'][50]))][var].median().values
                q_01 = ds.loc[dict(time = (targettime), chla_measure = name, quantile=(ds['quantile'][ 1]))][var].median().values
                scale  =  (refmed - ref01) / (med - q_01)
                offset =  refmed - med*scale            
                ds[var+'_offset'].loc[dict(chla_measure=name)] = offset
                ds[var+'_scale' ].loc[dict(chla_measure=name)] = scale
    return(ds)
    



# ---------------------------------------------------------------------------------------------
def read_dataset(filename):
    ds   = xr.open_dataset(filename).load(); ds.close()
    return(ds)

# ---------------------------------------------------------------------------------------------
def calc_statistics(ds,placename):
    q_values     = np.arange(0,1.01,0.01)
    tss_results  = ds.tss.quantile (q_values,dim=('x','y'))
    chla_results = ds.chla.quantile(q_values,dim=('x','y'))

    wq_results = xr.Dataset(
        data_vars =   None,
        coords     = {
                  'time': ds.time,
                  'place' : [placename],
                  'tss_measure' : ds.tss_measure,
                  'chla_measure': ds.chla_measure,
                  'quantile'    : q_values
                 },
        )
#    wq_results.coords['place'] = placename
    wq_results['tss']  = tss_results.expand_dims('place')
    wq_results['chla'] = chla_results.expand_dims('place')
    return(wq_results)


# ---------------------------------------------------------------------------------------------
def merge_results(log,filename,pathname,overwrite=True):
    suffix = '.nc'
    start = -1
    fname = pathname+filename+suffix 
    if not overwrite : 
        if os.path.exists(fname) :
            dsr = read_dataset(fname)
            start = 0
    for i in log.index.to_list() :
        fname = pathname+filename+'_'+log.loc[i,'PlaceName']+suffix
        if os.path.exists(fname) :
            if  start < 0 :
                start = i
                dsr   = read_dataset(fname)
            else :
                dsr=dsr.combine_first(read_dataset(fname))
    dsr.to_netcdf(pathname+filename+suffix)
    return(dsr)    

# ---------------------------------------------------------------------------------------------
def save_results(wq_results ,
                 filename ,
                 directory='/home/jovyan/deafrica_water_quality/wq_results/'):
    # ---- write to a file --- 
    wq_results.to_netcdf(directory+filename)
    return()


# ---------------------------------------------------------------------------------------------
def read_dataset(filename):
    ds   = xr.open_dataset(filename).load(); ds.close()
    return(ds)
       
# ---------------------------------------------------------------------------------------------
# --- start a log file for progress
def open_logfile (places_dict,logfilename = 'log.csv' , overwrite='Append',include_list=[]):
    if not os.path.exists(logfilename) or overwrite==True:
        print('-----------------\n Initiating new log file \n---------------------')
        placelist  = []
        for place in places_dict.keys(): 
            if places_dict[place]['run'] == True or \
            place in include_list : placelist.append(place)
        progress_log = pd.DataFrame(data = None,)
        progress_log['PlaceName'] = placelist
        progress_log['Status'] = 0
        progress_log['RunNumber'] = int(0)
        progress_log['Year1'] = int(0)
        progress_log['Year2'] = int(0)
        progress_log.to_csv(logfilename)
        
    else : progress_log = pd.read_csv(logfilename)
    if overwrite == 'Append':
        print('appending')
        placelist  = []
        for place in places_dict.keys(): 
            if places_dict[place]['run'] == True or \
                place in include_list: 
                placelist.append(place)    
                if not place in list(progress_log.PlaceName):
                    # --- rigmarole to insert a new row by copying penultimate one
                    progress_log.loc[len(progress_log)] = progress_log.loc[len(progress_log)-1]
                    progress_log.loc[len(progress_log)-1,'Status'] = 0
                    progress_log.loc[len(progress_log)-1,'PlaceName'] = place
                
        
    return(progress_log)

# ---------------------------------------------------------------------------------------------
def set_clearwater(ds):   
#    ds['clearwater'] = xr.where(np.isnan(ds.agm_fai),xr.where(ds.water_5yr==1,True,False),False)
    ds['clearwater'] = np.logical_and(np.isnan(ds['agm_fai'])  ,ds['water_5yr']  ==1)
    ds['clearwater'] = np.logical_and(         ds['clearwater'],ds['wofs_ann_water'])
    return(ds)

# ---------------------------------------------------------------------------------------------
def update_results(results,df_name,newcolname,dataseries):
    df = results[df_name]
    if newcolname in df.columns:  df = df.drop(labels=[newcolname],axis=1)
    df.insert(df.columns.size,newcolname,dataseries)
    return(results)

# ---------------------------------------------------------------------------------------------
def calculate_agm_count(ds):
    ds['agm_count']=('time','y','x'),np.zeros((ds.sizes['time'],ds.sizes['y'],ds.sizes['x']))
    for instrument in 'oli_agm','tm_agm','msi_agm':
        if instrument in ds.data_vars:
            ds['agm_count'] = ds['agm_count'] + ds[instrument+'_count'].where(ds[instrument]==True,0)
    return(ds)

# --- Initiate a results dictionary containing the analysis parameters and the results

# ---------------------------------------------------------------------------------------------
def initiate_results_dictionary(ds,parameters):
    results = {}
    results['parameters'] = parameters

    #The annual water area can be saved as a pandas data frame with years as indexes:
    results['annual_results'] = pd.DataFrame(
        {
         "time"    :ds.time,
         "year"    :pd.DatetimeIndex(ds.time).year,
         "area_km2":(ds.water_5yr.count(dim=('x','y'))*parameters['cell_area']).round(2)},     
        #index = pd.DatetimeIndex(ds_annual.time).year
        )
    for instrument in 'oli_agm','tm_agm','msi_agm':
        if instrument in ds.data_vars:
            results['annual_results'].insert(
                2,
                instrument,
                (ds[instrument])
                )

    results['annual_results'].insert\
            (
            results['annual_results'].columns.size,
            'total_geomedian_count',
            ds['agm_count'].mean(dim=('x','y')).round()
            )
    results['annual_results'].insert\
            (
            results['annual_results'].columns.size,
            'average_water_freq',
            (ds.wofs_ann_wetcount/ds.wofs_ann_clearcount).where(ds.water_5yr>0).mean(dim=('x','y'))
            )
    return(results)

# ---------------------------------------------------------------------------------------------
def store_parameters(placename,spacetime_domain, grid_resolution, cell_area, resampling_option, year1, year2): 
    y1,y2 = pd.DatetimeIndex([spacetime_domain['time'][0],spacetime_domain['time'][1]]).year[[0,1]]
    parameters = {}
    parameters['placename'] = placename
    parameters['xyt'] = spacetime_domain
    parameters['grid_resolution'], parameters['cell_area'], parameters['resampling_option'],parameters['year1'],parameters['year2'] = \
    grid_resolution,                   cell_area,                resampling_option,            y1,y2
    return(parameters)

# ---------------------------------------------------------------------------------------------
# --- save the analysis parameters ---
def log_update(log,placename,status,logfilename,verbose=False) :
    log.loc[log.PlaceName==placename,'Status']     = status
    log.to_csv(logfilename)
    if verbose : print('Status : '+str(status))
    return(log)

# ---------------------------------------------------------------------------------------------
# --- save the analysis parameters ---
def log_parameters(log,placename,parameters) :
    if True :
        log.loc[log.PlaceName==placename,'Year1']      = int(parameters['year1'])
        log.loc[log.PlaceName==placename,'Year2']      = int(parameters['year2'])
        log.loc[log.PlaceName==placename,'x0']          =    parameters['xyt']['x'][0]
        log.loc[log.PlaceName==placename,'x1']          =    parameters['xyt']['x'][1]
        log.loc[log.PlaceName==placename,'y0']          =    parameters['xyt']['y'][0]
        log.loc[log.PlaceName==placename,'y1']          =    parameters['xyt']['y'][1]
        log.loc[log.PlaceName==placename,'t0']          =    parameters['xyt']['time'][0]
        log.loc[log.PlaceName==placename,'t1']          =    parameters['xyt']['time'][1]        
        log.loc[log.PlaceName==placename,'CellArea']    =    parameters['cell_area']
        log.loc[log.PlaceName==placename,'Resampling']  =    parameters['resampling_option']
        return(log)
    else : return(parameters , logfile)

# ---------------------------------------------------------------------------------------------
def check_instrument_dates(instruments_to_use,year1,year2,verbose=True):
    # --- this function changes the values in the dictionary. Those changes apply globally, it seems. 
    #---  cross-check against the years for which the analysis is going to be run.
    instrument_dates = {
        'oli_agm'  : [2013,2024],
        'oli'      : [2013,2025],
        'msi_agm'  : [2017,2024],
        'msi'      : [2017,2025],
        'wofs_ann' : [1990,2024],
        'wofs_all' : [1990,2024],
        'tm_agm'   : [1990,2012],
        'tm'       : [1990,2023],
        'tirs'     : [2000,2025],
        }

    for name in list(instruments_to_use.keys()):
        if verbose: print(name, instruments_to_use[name]['use'])
        if not (instrument_dates[name][1] >= int(year1) and instrument_dates[name][0] <= int(year2)):
            if verbose: print('instrument ',name,' has date ranges ',instrument_dates[name][0],instrument_dates[name][1],' outside of ',year1,year2)
            instruments_to_use[name]['use'] = False
    return(instruments_to_use)




    