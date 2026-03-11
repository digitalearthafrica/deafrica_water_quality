# functions requred by for WP2.1
# trying to not overlap with _WQ_functions in WP1.2
# based on _WP2.2_functinos
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

# --- default coefficients from WP2.2. ----
def calibrate(data,varname):
    # assume that we have a quadratic calibration model that could be measurement-specific
    coeffs  = {'tss'        : [-2.0559,.5824,0],
              'chla'        : [-27.342,2.0894,0],
              'chla_tc2_msi': [0,1,0]    }
    if varname in coeffs.keys() :
        coeffs=coeffs[varname]
        return(
            coeffs[0] + \
            coeffs[1]*data + \
            coeffs[2]*pow(data,2) 
              )
    else :
        print(f'{var} not found in the default coefficients list')
        return(data) 


# ------------------------------------------------------------
def determine_water_area (ds,ds_annual):
    year         = pd.DatetimeIndex(ds['time']).year[int(ds.time.size/2)]  # the year at the centre of the monitoring interval
    water_5yr    = ds_annual.water_5yr     .sel(time=str(year),method='nearest').squeeze()   
    water_annual = ds_annual.wofs_ann_water.sel(time=str(year),method='nearest').squeeze()   
    # a data array with xy shape
    water_pixel_count = water_5yr.count()

    # --- calculate the proportion of observations that are present and valid, at each time-step
    ds['data_coverage']   =  (ds.where(~np.isnan(ds.qa_score)).where(water_5yr == 1).qa_score.count(dim=('x','y'))/water_pixel_count)

    # --- calculate the proportion of valid observations that pass QA
    ds['data_qapass']     =  (ds.where(ds.qa_score >0)        .where(water_5yr == 1).qa_score.count(dim=('x','y'))/water_pixel_count) \
        / ds['data_coverage']

    # --- save the water pixel count as a variable in the dataset ...
    ds['water_pixel_count'] = water_pixel_count

    # --- save the water mask within the dataset 
    ds['water_5yr']    = water_5yr
    ds['water_ann'] = water_annual
    return(ds)
 


# -----------------------------------------------------------------------
# --- a function to make nice step-plot data; we assume that we're working with time data and that the step intervals are even
def make_stepplot_data(tdata,ydata):
#    delta = ((np.datetime64(tdata[1])- np.datetime64(tdata[0]))/2)
    delta = (((tdata[1])- (tdata[0]))/2)
    steps = np.array([], dtype='datetime64')
    values = []
    for t in tdata:
        value  = ydata[tdata==t]
        tminus = t - delta
        tplus  = t + delta
        steps  = np.append(steps, [tminus.values,tplus.values])
        values = np.append(values,[value ,value])
    return(steps,values)

# ---------------------------------------------------------------------------------------------
def calculate_agm_count(ds):
    ds['agm_count']=('time','y','x'),np.zeros((ds.sizes['time'],ds.sizes['y'],ds.sizes['x']))
    for instrument in 'oli_agm','tm_agm','msi_agm':
        if instrument in ds.data_vars:
            ds['agm_count'] = ds['agm_count'] + ds[instrument+'_count'].where(ds[instrument]==True,0)
    return(ds)


# ------------------------------------------------------------------------------------------
# a function to calcultate the NDVI; for non-geomedian datasets at this point since its already covered for geomedians
# This function calculates the NDVI for a designated instrument.

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
    for inst in ('tm','oli','msi'):
        if inst  in ds.data_vars:    
            ds['ndvi'].loc[ds[inst] ==True] = NDVI(ds.where(ds.ndvi.loc[ds[inst]== True]),inst )
    # to avoid outliers (in oli) also truncate to the valid range of ndvi, i.e., less than or equal to 1.0
    ds['ndvi'] = ds.ndvi.where(ds.ndvi > threshold).where(watermask==1).where(ds.qa_score >0).where(ds.ndvi<=1.0) 
    return(ds)

# ---------------------------------------------------------------------------------------------
def calc_fai(ds,watermask,threshold = 0.0):
    ds['fai'] = ('time','y','x'),np.zeros([ds.qa_score.sizes['time'],ds.qa_score.sizes['y'],ds.qa_score.sizes['x']])*np.nan
    for inst in ('tm','oli','msi'):
        if inst  in ds.data_vars:    
            ds['fai'].loc[ds[inst] ==True] = FAI(ds.where(ds.fai.loc[ds[inst]== True]),inst )

    ds['fai'] = ds.fai.where(ds.fai > threshold).where(watermask==1).where(ds.qa_score >0) 
    return(ds)

# ---------------------------------------------------------------------------------------------
# --- a function that adds boolean variables that retain the source instrument, and a qa_score variable. 
# Stil Testing that this behaves correctly when the qa_score is pre-set using the agm tests ....
def add_vars_and_combine_datasets(data_list):
    # --- setting qa_score to nan, 0 or 1
    for instrument in data_list.keys():
        # --- select the relevant dataset ---
        ds             = data_list[instrument]
        ds[instrument] = ('time'), np.ones(ds.time.sizes['time']).astype('bool')                    # --- a boolean variable with a value of 'True' ---
        if not 'qa_score' in ds.data_vars :
            ds['qa_score'] = ('time','y','x'), np.zeros([ds.sizes['time'],ds.sizes['y'],ds.sizes['x']]) # --- a "qa_score" variable and set to zero ----

        # --- attempt to set the value of the qa_score, based on the pixel quality data provided with the data.  
        if instrument == 'tm':  
            ds['qa_score'] = xr.where(ds.tm,
                                      xr.where(np.isnan(ds.tm_qa),np.nan,ds.tm_qa),ds.qa_score)   # --- sets the qa_score to nan where there is nodata (should not happen)
            # --- frankly, this is pretty flaky; I don't trust that the data are consistenly labeled and with TM I am picking specific values
            # ---- values of 5504 should be correct, but 5503 and 5505 also appear. 
        
            ds['qa_score'] = xr.where(np.isin(ds.tm_qa,[5503,5504,5505,5440]),ds.qa_score,0) #DEA mess around with the pixel values so that they sort of conform to OPI, but not really.
            ds['qa_score'] = xr.where(ds.tm_qa==1 , np.nan , ds.qa_score) #values of 1 are 'fill', set to nan

        if instrument == 'oli': 
            ds['qa_score'] = xr.where(ds.oli,
                                      xr.where(np.isnan(ds.oli_qa),np.nan,ds.oli_qa),ds.qa_score)  # --- sets the qa_score to nan where there is no coverage
            ds['qa_score'] = xr.where(ds.oli_qa<=21952 , ds.qa_score,0) 
            ds['qa_score'] = xr.where(ds.oli_qa==1 , np.nan , ds.qa_score) #values of 1 are 'fill', set to nan
            # oli qa scores are found here : https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
            # 21952 corresponds to a clear pixel over water. This is the base for use here.

        if instrument == 'msi': 
            ds['qa_score'] = xr.where(ds.msi,
                                      xr.where(np.isnan(ds.msi_qa),np.nan,ds.msi_qa),ds.qa_score)   # --- sets the qa_score to nan where there is no coverage (should not happen)
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
    if 'crs' in parameters.keys():
        crs = parameters['crs']
    else: crs = 'epsg:6933'
        
    instruments,measurements,rename_dict = instruments_list(instruments_to_use) 
    datasets = {}
    dc = datacube.Datacube(app='build_agm_dataset')
    for instrument in list(instruments_to_use.keys()):
        if instruments_to_use[instrument]['use'] :
            if verbose : print('loading data for ',instrument,'...')
            datasets[instrument] = dc.load(product=(products[instrument]),
                                 **spacetime_domain,
                                 **{'measurements': measurements[instrument]},
                                 output_crs=crs,
                                 resolution=parameters['grid_resolution'],
                                 align=(0,0),
                                 resampling=parameters['resampling_option'],
                                 )
    
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
# a function to build a dataset from the datacube, for each key sensor (msi, oli, etm)
# returns a dictionary of datasets
# based on the FAI notebook I think, or maybe WP2.1

def build_dataset (spacetime_domain,
                   instruments_to_use, 
                   products, 
                   measurements, 
                   grid_resolution,
                   resampling_option,
                   rename_dict,
                   crs='epsg:6933',
                   verbose=True) :

    dc = datacube.Datacube(app='WP22_C_calibration_dataset')

    data_list = {}
    if instruments_to_use['oli']['use']:
        if verbose : print('building the oli dataset...')
        instrument = 'oli'
        # --- load oli data
        # Load available data from all three Landsat satellites
        dc = datacube.Datacube(app='building_dataset')
        ds_oli = dc.load(product=(products[instrument]),
                                 **spacetime_domain,
                                 **{'measurements': measurements[instrument]},
                                 output_crs=crs,
                                 group_by ='solar_day',
                                 resolution=grid_resolution,
                                 align=(0,0),
                                 resampling=resampling_option,)
        # --- re-name the variables for the sake of sanity --- 
        ds_oli = rename_vars_robust(ds_oli,rename_dict['oli'])
        # --- set zeros to nans and re-scale 
        for var in ds_oli.data_vars:
            ds_oli[var] = xr.where(ds_oli[var]>0,ds_oli[var],np.nan)
            if not var == 'oli_qa':
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
                                 output_crs=crs,
                                 group_by ='solar_day',
                                 resolution=grid_resolution,
                                 align=(0,0),
                                 resampling=resampling_option,)
        # --- re-name the variables for the sake of sanity --- 
        ds_msi = rename_vars_robust(ds_msi,rename_dict['msi'])
        # --- set zeros to nans and re-scale 
        for var in ds_msi.data_vars:
            ds_msi[var] = xr.where(ds_msi[var]>0,ds_msi[var],np.nan)
            if not var == 'msi_qa':
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
                                 output_crs=crs,
                                 group_by ='solar_day',
                                 resolution=grid_resolution,
                                 align=(0,0),
                                 resampling=resampling_option,)

        # --- re-name the variables for the sake of sanity --- 
        ds_tm = rename_vars_robust(ds_tm,rename_dict['tm'])
    

        # --- set zeros to nans and re-scale 
        for var in ds_tm.data_vars:
            ds_tm[var] = xr.where(ds_tm[var]>0,ds_tm[var],np.nan)
            if not var == 'tm_qa':
                ds_tm[var] = ((ds_tm[var] * 0.0000275) - 0.2) * 10000

        data_list['tm'] = ds_tm
        if verbose : print('... done.')

    #yet to switch this on...
    if instruments_to_use['tirs']['use'] :
        test = test
        if verbose : print('building the tirs dataset ...')
        instrument = 'tirs'
        # --- load tirs data
        ds_tirs = dc.load(product=(products[instrument]),
                                 **spacetime_domain,
                                 **{'measurements': measurements[instrument]},
                                 output_crs=crs,
                                 group_by ='solar_day',
                                 resolution=grid_resolution,
                                 align=(0,0),
                                 resampling=resampling_option,)

        # --- re-name the variables for the sake of sanity --- 
        ds_tirs = rename_vars_robust(ds_tirs,rename_dict['tirs'])

        # adjust and qa the temperature 
        ds_tirs['tirs_st']    = (ds_tirs.tirs_st * 0.00341802 + 149.0) - 273.15
        ds_tirs['tirs_st_qa'] = ds_tirs['tirs_st_qa'] * 0.01    # -- uncertainty in kelvin 
        ds_tirs['tirs_emis']  = ds_tirs['tirs_emis' ] * 0.0001  # -- emissivity fraction
        ds_tirs['tirs_st']    = xr.where(ds_tirs['tirs_st'] > 0,
                                xr.where(ds_tirs['tirs_st_qa'] < 5,
                                     xr.where(ds_tirs['tirs_emis']> 0.95,
                                              ds_tirs['tirs_st'],
                                              np.nan),
                                     np.nan),
                                np.nan)

        data_list['tirs']=ds_tirs
        if verbose : print('... done.')

    print('... instrument datasets complete\n')
    return(data_list)


# ---------------------------------------------------------------------------------------------
def read_dataset(filename):
    ds   = xr.open_dataset(filename).load(); ds.close()
    return(ds)


# ---------------------------------------------------------------------------------------------
def set_clearwater(ds, fai_var, water_5yr_var, water_annual_var):   
#    ds['clearwater'] = xr.where(np.isnan(ds.agm_fai),xr.where(ds.water_5yr==1,True,False),False)
    ok = True
    if not type(ds) == type(xr.Dataset())       : ok = False
    if not fai_var              in ds.data_vars : ok = False
    if not water_5yr_var        in ds.data_vars : ok = False
    if not water_annual_var == None:
        if not water_annual_var in ds.data_vars : ok = False
    if not ok:
        print('set_clearwater -> invalid arguments, ending (dataset unchanged)')
        return(ds)
    ds['clearwater']     = np.logical_and(np.isnan(ds[fai_var])  ,ds[water_5yr_var]  ==1)
    if water_annual_var != None:
        ds['clearwater'] = np.logical_and(ds['clearwater'],ds[water_annual_var]>0)
    return(ds)








