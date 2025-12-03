#!/usr/bin/env python 
# coding: utf-8
import numpy  as np
import xarray as xr
import matplotlib.pyplot as plt
import gc   # garbage collection
import pandas as pd

# --- Add boolean variables that indicate which instruments are available in the geomedian data series for a given year. For earlier years tm is available. 
# --- assume that the 'smad' will always be available and use this as an indicator 
# --- If the instrument is never available during the time series, the boolean variable is not added.
def detect_geomedian_instruments (ds_annual):
    for inst in 'oli_agm','tm_agm','msi_agm':
        if inst+'_smad' in ds_annual.data_vars:
            ds_annual[inst] = ('time'), np.zeros(ds_annual.sizes['time'],dtype ='bool')
            ds_annual[inst] = ~np.isnan(ds_annual[inst+'_smad'].mean(dim=('x','y')))
    return ds_annual


def run_OWT(ds,verbose=False):
    OWT_vector_data = create_OWT_response_models(agm=True)
    suffix = '_agm'
    for instrument in 'msi','oli','tm':
        if instrument+suffix in ds.data_vars:
            OWT_vectors = OWT_vector_data[instrument]    # the spectral reflectances specific to this sensor
            OWT_data    = OWT(ds.where(ds.clearwater==1),
                          instrument,
                          OWT_vectors,
                          agm=True,
                          test=False, 
                          verbose=verbose) 
            varname = instrument+'_agm_owt'
            if varname in ds.data_vars:
                ds[varname] = xr.where(~np.isnan(OWT_data),OWT_data,ds[varname])
            else : 
                ds[varname] = OWT_data
            del OWT_data
    silent  = gc.collect()
    return(ds)


# This function calculates a running 5-year wofs frequency on the dataset
# --- The start and end values are calculated on the nearest 5 year window; otherwise the centre of the window
# --- *** Change needed to put the window as the interval leding upto the year in question to ensure future compatibility futher processing ** 
# --- If the duration of the geomedian series is less than 5 years the frequency is calculated on that shorter time series. 
# ! assuming the shape of the array is time,y,x and using array-based indexing rather than dataset labels.

def WOFS_5_year_frequency(ds, test = True):
    clearcount = np.zeros(ds.wofs_ann_freq.shape,dtype = 'int')
    wetcount   = np.zeros(ds.wofs_ann_freq.shape,dtype = 'int')
    freq       = np.zeros(ds.wofs_ann_freq.shape,dtype = 'float')
    
    length = ds.time.size
    window = 5
    # --- I am sure there is a simpler formulation, but this works! ---
    for i in np.arange(0,length):
        start = np.max([0,i-int(window/2)])
        end   = np.min([length,start+5])    
        start = np.min([start,length-5])
        start = np.max([start,0])
        if test: print(i,start,end)
    
        clearcount[i,:,:] = ds.wofs_ann_clearcount[start:end,:,:].sum(dim=('time'))
        
        wetcount[i,:,:]   = ds.wofs_ann_wetcount[start:end,:,:].sum(dim=('time'))
        if test : print(start,end,clearcount[i,:,:].sum(),clearcount.sum(),wetcount[i,:,:].sum(),wetcount.sum())

    np.divide(wetcount, clearcount, out = freq,where=(clearcount>0))

    if test : print(np.nanmean(freq),np.sum(wetcount), np.sum(clearcount))
    ds['wofs_5yr_freq']             = ds.wofs_ann_freq.dims, freq 
    ds['wofs_5yr_clearcount']       = ds.wofs_ann_freq.dims, clearcount 
    ds['wofs_5yr_wetcount']         = ds.wofs_ann_freq.dims, wetcount 
    if test : print(ds.wofs_5yr_freq.mean().item(),np.nanmean(freq),ds.wofs_5yr_wetcount.sum().item(),np.sum(wetcount),ds.wofs_5yr_clearcount.sum().item(), np.sum(clearcount))
    
    return(ds)
    


#----------------------------------------------------------------------
#  A function to calculate the Trophic State Index (TSI) from ChlA estimates. 
#  For completeness, I include the data array sourced from the SDG 6.6.1 web site, which 
#  gives the TSI as a step function. However it is just a log transfrom of the ChlA, so a much neater 
#  implementation is to derive the TSI as a continuous variable. 
#  The pandas data frame part is not essential; just convenient at the time.
def trophic_state(da):
    # source data
    data = np.array((0,0.04,10,0.12,20,0.34,30,0.94,40,2.6,50,6.4,60,20,70,56,80,154,90,427,100,1183)).reshape(11,2)
    df = pd.DataFrame(data=None,
                  columns = ['TSI','ChlA'],
                 )
    df['TSI' ] = data[:,0].astype('int')
    df['ChlA'] = data[:,1]

    # derive the log-linear model; in practice these parameters will not change, slope =22.4622417, intercept = 30.9129
    slope,intercept = np.polyfit(np.log10(df.ChlA),df.TSI,1) 
    min        = 0.01
    da[da<min] = min
    tsi = np.log10(da) * slope + intercept + 0.53   #the 0.53 ensures that the lowest values match. A nicety.
    
    tsi[tsi<0] = 0
    tsi[tsi>100] = 100
    tsi[np.isnan(da)]=np.nan
    return(tsi.round(2))  


# --------------------------------------------------------------------------------------------------------
def WQ_vars(ds,
            algorithms,              # a dictionary of  algorithms instruments, and bands
            instruments,             # a dictionary of the instruments that have been used to build the dataset
            new_dimension_name=None, # 'tss_measures', or 'chla_measures', or None
            new_varname=None,        # 'tss' or 'chla' - the name of the variable that will be calculated
            verbose=True,
            test=False):  
    
    # ---- Run the TSS/TSP algorithms applying each algorithm to the instruments and band combinations set in the dictionary,
    #      checking that the necessary instruments are in the dataset
    
    if test: print(' \nRunning  WQ algorithms for:\n',list(algorithms.keys()))    
    wq_varlist =  []
    for alg in algorithms.keys():
        if test: print("\n algorithms = ",alg)
        for inst in list(algorithms[alg].keys()): 
            if test : print('instrument = ',inst)
            params = algorithms[alg][inst]
            
            if inst in list(instruments.keys()): 
                if test : print('instrument found',inst)
                if inst == 'msi_agm' and alg == 'ndci_nir_r' :  #special case here as options are possible
                    for option in params.keys():
                        opparams = params[option]
                        function = opparams['func']
                        ds[opparams['wq_varname']] = function(ds,**opparams['args'],verbose=verbose)
                        wq_varlist = np.append(wq_varlist,opparams['wq_varname'])
                else: 
                    if test : print('instrument = ',inst,' new varname = ',params['wq_varname'])
                    function = params['func']
                    ds[params['wq_varname']] = function(ds,**params['args'],verbose=verbose)
                    wq_varlist = np.append(wq_varlist,params['wq_varname'])
    
        #If relevant arguments are provided, then create a dimension for the data variables and move them into it
    if (new_dimension_name!=None and new_varname!=None):
        new_dim_labels = list(wq_varlist)
        if test : 
            print('\nnew dimension labels (variables):\n',new_dim_labels,'\nnew dimension name: ',new_dimension_name)
  
        ds=ds.assign_coords({new_dimension_name:new_dim_labels})
        ds[new_varname] = (('time','y','x',new_dimension_name),\
        np.zeros((ds.time.size,ds.y.size,ds.x.size,ds[new_dimension_name].size)))
        if test : print("\n",ds[new_varname])
        if test : print("\n--->  ",ds[new_dimension_name].values,"  <---- new dimension values\n")
        if test : print("\n--->  ",new_varname,"  <---- new variable name (should be tss or chla\n")
        #move the tss data into the new dimension:
        for name in list(ds[new_dimension_name].values):
           if test: 
              print("\nnew_dimension_name -->",new_dimension_name)
              print("\nname               -->",name)
              print("\nnew_varname        -->",new_varname)
           ds[new_varname].loc[:,:,:,name] = ds[name]
        
        #drop  the old variables to keep it tidy:
        if True: ds= ds.drop_vars(list(ds[new_dimension_name].values))
    return ds,wq_varlist

    
# ------------------------------------------------------------------------------------------------------------------------------------
# --- Calculate NDVI values for annual geomedians instrument ----
#     NDVI is calculated for each instrument in the series (tm, oli, msi).
#     The overall NDVI is a weighted mean (weighted by the number of observations in each geomedian)

def geomedian_NDVI(ds_annual,water_mask,test=False):
    ndvi_bands = {}
    ndvi_bands['tm']  = ['tm04','tm03']
    ndvi_bands['oli'] = ['oli05','oli04']
    ndvi_bands['msi'] = ['msi8a','msi04']

    # --- these values are used as a simple adjustment of the ndvi values to maximise comparability
    #     In practice over a few years the NDVI distributions are remakably consistent, but in any given year they are quite divergent.
    
    reference_mean = {
        'ndvi' : {'msi_agm': 0.2335, 'oli_agm' : 0.2225, 'tm_agm': 0.2000},
        }
    threshold = {
        'ndvi' : {'msi_agm': 0.05, 'oli_agm' : 0.05, 'tm_agm': 0.05},
        }
    count = 0
    for inst in list(('msi','oli','tm')):
        inst_agm = inst+'_agm'
        if inst_agm in ds_annual.data_vars: 
            scale = reference_mean['ndvi']['msi_agm'] / reference_mean['ndvi'][inst_agm]
            # --- calculate the NDVI for this sensor ---
            ndvi_data = \
                ((ds_annual[ndvi_bands[inst][0]+'_agm'] - ds_annual[ndvi_bands[inst][1]+'_agm']) / \
                 (ds_annual[ndvi_bands[inst][0]+'_agm'] + ds_annual[ndvi_bands[inst][1]+'_agm'])) * scale


            # --- set nans to zero, and also in the agm_count variable in the dataset which is more logically zero rather than nan
            ndvi_data                    = np.where(~np.isnan(ndvi_data),ndvi_data,0)
            ds_annual[inst_agm+'_count'] = xr.where(~np.isnan(ds_annual[inst_agm+'_count']),ds_annual[inst_agm+'_count'],0)

            # --- this step must be done before thresholding for the instrument, since that brings in nans
            if count == 0: 
                mean_ndvi = ndvi_data * ds_annual[inst_agm+'_count']
                agm_count =             ds_annual[inst_agm+'_count']
            else :    
                mean_ndvi = mean_ndvi + ndvi_data * ds_annual[inst_agm+'_count']
                agm_count = agm_count + ds_annual[inst_agm+'_count']
            count = count + 1

            # --- trim the ndvi values back to relevant areas and values
            ndvi_data = np.where(ndvi_data > threshold['ndvi']['msi_agm'],ndvi_data,np.nan)
            ndvi_data = np.where(~np.isnan(water_mask),ndvi_data,np.nan)   # new code with mask 

            # --- this retains an ndvi for each instument, but that is not essential
            ds_annual[inst_agm+'_ndvi'] = (ds_annual.wofs_ann_freq.dims),ndvi_data

    # --- divide by the total count to get the actual mean, then trim back to relevant values and areas
    mean_ndvi = mean_ndvi / agm_count
    mean_ndvi = np.where(mean_ndvi > threshold['ndvi']['msi_agm'] ,mean_ndvi,np.nan)
    mean_ndvi = np.where(~np.isnan(water_mask)                    ,mean_ndvi,np.nan)

    ds_annual['agm_ndvi'] = (ds_annual.wofs_ann_freq.dims), mean_ndvi
    return(ds_annual)

            
    
# ------------------------------------------------------------------------------------------------------------------------------------
#     Values from different sensors are standardised using mean values developed empirically, with MSI taken as the reference becasue there are more msi data. 
#     The FAI function needs to kmow the instrument because the bands used are instrunent-specific
#     Revised version:
#     - addresses a bug in taking the averaged fai value, 
#     - uses an externally supplied water mask rather than an internal test,
#     - applies a threshold for valid fai values (i.e., above zero)
#     - includes multiple test l

def geomedian_FAI (ds_annual,water_mask,test=False):
    reference_mean = {
        'fai'  : {'msi_agm': 0.0970, 'oli_agm' : 0.1015, 'tm_agm': 0.0962},
        }
    threshold = {
        'fai' : {'msi_agm': 0.05   , 'oli_agm' : 0.05   , 'tm_agm': 0.05},
        }
    if test : print('starting')
    count = 0   
    for inst in list(('msi','oli','tm')):
        inst_agm = inst+'_agm'
        if inst_agm in ds_annual.data_vars: 
            scale = reference_mean['fai']['msi_agm'] / reference_mean['fai'][inst_agm]

            # --- calculate the FAI for this sensor
            fai_data = FAI(ds_annual,inst_agm,test = False) * scale

            # --- set nans to zero, and also in the agm_count variable
            fai_data                     = np.where(~np.isnan(fai_data),fai_data,0)
            ds_annual[inst_agm+'_count'] = xr.where(~np.isnan(ds_annual[inst_agm+'_count']),ds_annual[inst_agm+'_count'],0)

            # --- this step must be done before thresholding for the instrument, since that brings in nans
            if count == 0: 
                mean_fai  =  fai_data * ds_annual[inst_agm+'_count']
                agm_count =             ds_annual[inst_agm+'_count']
                if test: print('\n fai and agm count initial',mean_fai,agm_count)  
            else :    
                mean_fai  = mean_fai  + fai_data * ds_annual[inst_agm+'_count']
                agm_count = agm_count + ds_annual[inst_agm+'_count']
                if test: print('\n fai and agm count compounded',mean_fai,agm_count)
            count = count + 1

            # --- trim the fai values back to relevant areas and values
            fai_data = np.where(fai_data > threshold['fai']['msi_agm'],fai_data,np.nan)
            
            # fai_data = np.where(ds_annual.wofs_5yr_freq > WAFT,fai_data,np.nan) 
            fai_data = np.where(~np.isnan(water_mask),fai_data,np.nan)   # new code with mask 

            # --- this retains an fai for each instument, but that is not essential
            ds_annual[inst_agm+'_fai'] = (ds_annual.wofs_ann_freq.dims),fai_data
            if test: print('\n','instrument fai result:',ds_annual[inst_agm+'_fai'])

    # --- divide by the total count to get the actual mean, then trim back to relevant values and areas
    mean_fai = mean_fai / agm_count
    if test: print('\n','agm fai result:',mean_fai)
    mean_fai = np.where(mean_fai > threshold['fai']['msi_agm'] ,mean_fai,np.nan)
    if test: print('\n','agm fai result after thresholding:',mean_fai)
    mean_fai = np.where(~np.isnan(water_mask)               ,mean_fai,np.nan)
    if test: print('\n','agm fai result after water mask:',mean_fai)

    ds_annual['agm_fai'] = (ds_annual.wofs_ann_freq.dims), mean_fai

    return(ds_annual)

   
    
# ------------------------------------------------------------------------------------------------------------------------------------
# FAI algorithm depends on knowledge of the central wavelengths of the red, nir, and swir sensors. I include  a dictionary of those values here to keep the function self-contained
def FAI (ds, instrument, test=False):
    inst_bands = ib = {}
    ib['msi']     = {  'red' : ('msi04',     665),          'nir' : ('msi8a',     864),          'swir': ('msi11',     1612)            }
    ib['msi_agm'] = {  'red' : ('msi04_agm', 665),          'nir' : ('msi8a_agm', 864),          'swir': ('msi11_agm', 1612)            }
    ib['oli']     = {  'red' : ('oli04',    (640 + 670)/2), 'nir' : ('oli05',    (850 + 880)/2), 'swir': ('oli06',    (1570 + 1650)/2)  }
    ib['oli_agm'] = {  'red' : ('oli04_agm',(640 + 670)/2), 'nir' : ('oli05_agm',(850 + 880)/2), 'swir': ('oli06_agm',(1570 + 1650)/2)  }
    ib['tm']      = {  'red' : ('tm03',     (630 + 690)/2), 'nir' : ('tm04',     (760 + 900)/2), 'swir': ('tm05',     (1550 + 1750)/2)  }
    ib['tm_agm']  = {  'red' : ('tm03_agm', (630 + 690)/2), 'nir' : ('tm04_agm',(760 + 900)/2),  'swir': ('tm05_agm', (1550 + 1750)/2)  }
    if not instrument in inst_bands.keys():
        print('! -- invalid instrument, FAI will be calculated as zero --- !')
        return(0)
    red, l_red   = ib[instrument]['red'][0], ib[instrument]['red'] [1]
    nir, l_nir   = ib[instrument]['nir'][0], ib[instrument]['nir'] [1]
    swir,l_swir  = ib[instrument]['swir'][0],ib[instrument]['swir'][1]

            
    # --- final value is scaled by 10000 to reduce to a value typically in the range of 0-1 (this assumes that our data are scaled 0-10,000)     
    return((ds[nir] - ( ds[red] + ( ( ds[swir] - ds[red] ) * ( ( l_nir - l_red ) / ( l_swir - l_red ) ) ) )) / 10000)

# ------------------------------------------------------------------------------------------------------------------
# A function to set the dictionary of instruments, bands, algorithms, and  functions ----------------------- 
# --- revised to do away with the 'r' suffix
# --- revised to remove duplication, with the suffix argument used to decide if the analysis is annual geomedian or other

def set_wq_algorithms(suffix=''):
    s = suffix
    ndci_nir_r   =  { 
                "msi"+s   : {'54' : {'func': NDCI_NIR_R, "wq_varname" : 'ndci_msi54'       ,'args' : {"NIR_band" : 'msi05'+s , "red_band" :'msi04'+s}},
                               '64' : {'func': NDCI_NIR_R, "wq_varname" : 'ndci_msi64'     ,'args' : {"NIR_band" : 'msi06'+s , "red_band" :'msi04'+s}},
                               '74' : {'func': NDCI_NIR_R, "wq_varname" : 'ndci_msi74'     ,'args' : {"NIR_band" : 'msi07'+s , "red_band" :'msi04'+s}}},
                "tm"+s    : {'func': NDCI_NIR_R, "wq_varname" : 'ndci_tm43'            ,'args' : {"NIR_band" : 'tm04'+s  , "red_band" : 'tm03'+s}},
                "oli"+s   : {'func': NDCI_NIR_R, "wq_varname" : 'ndci_oli54'           ,'args' : {"NIR_band" : 'oli05'+s , "red_band" :'oli04'+s}}
                }

    chla_toming  = { #looks more like a tss indicator!
                    "msi"+s     : {'func': ChlA_Toming, "wq_varname" : 'chla_toming_msi' , 'args' : {"band5" : 'msi05'+s , "band4":'msi04'+s , 'band6' : 'msi06'+s }}
                    }
    chla_3bda    = { 
                    "msi"+s     : {'func': ChlA_3BDA, "wq_varname" : 'chla_3bda_msi' , 'args' : {"blue_band" : 'msi02'+s , "red_band":'msi04'+s , 'green_band' : 'msi03'+s }},
                    "oli"+s     : {'func': ChlA_3BDA, "wq_varname" : 'chla_3bda_oli' , 'args' : {"blue_band" : 'oli02'+s , "red_band":'oli04'+s , 'green_band' : 'oli03'+s }},
                    "tm"+s      : {'func': ChlA_3BDA, "wq_varname" : 'chla_3bda_tm'  , 'args' : {"blue_band" : 'tm01'+s  , "red_band":'tm03'+s  , 'green_band' : 'tm02'+s  }}
                    }
    chla_tebbs  = {  #this is just a ratio of two bands
                    "msi"+s     : {'func': ChlA_Tebbs, "wq_varname" : 'chla_tebbs_msi' , 'args' : {"NIR_band" : 'msi8a'+s , "Red_band":'msi04'+s  }},
                    "oli"+s     : {'func': ChlA_Tebbs, "wq_varname" : 'chla_tebbs_oli' , 'args' : {"NIR_band" : 'oli05'+s , "Red_band":'oli04'+s  }},
                    "tm"+s      : {'func': ChlA_Tebbs, "wq_varname" : 'chla_tebbs_tm'  , 'args' : {"NIR_band" : 'tm04'+s  , "Red_band":'tm03'+s  }}
                    }
    chla_meris2b = {
                    "msi"+s     : {'func': ChlA_MERIS2B, "wq_varname" : 'chla_meris2b_msi'     ,'args' : {"band_708" : 'msi05'+s     , "band_665":'msi04'+s    }}
                    }

    chla_modis2b = {
                    "msi"+s  : {'func': ChlA_MODIS2B, "wq_varname" : 'chla_modis2b_msi'        , 'args' : {"band_748" : 'msi06'+s , "band_667":'msi04'+s}},
                     "tm"+s  : {'func': ChlA_MODIS2B, "wq_varname" :  'chla_modis2b_tm'        , 'args' : {"band_748" :  'tm04'+s , "band_667": 'tm03'+s}}
                    }

    ndssi_rg     = {
                "msi"+s     : {'func': NDSSI_RG,     "wq_varname" : 'ndssi_rg_msi'            ,'args'   : { "red_band":'msi04'+s    , "green_band":'msi03'+s    }},
                "oli"+s     : {'func': NDSSI_RG,     "wq_varname" : 'ndssi_rg_oli'            ,'args'   : { "red_band":'oli04'+s    , "green_band":'oli03'+s    }},
                "tm"+s      : {'func': NDSSI_RG,     "wq_varname" : 'ndssi_rg_tm'             ,'args'   : { "red_band":'tm03'+s     , "green_band": 'tm02'+s    }}
                }

    ndssi_bnir   = {   
                #"msi"     : {'func': NDSSI_BNIR,   "wq_varname" : 'ndssi_bnir_msi'         ,'args' : { "NIR_band":'msi8a'     , "blue_band":'msi02_agm'}},
                "oli"+s     : {'func': NDSSI_BNIR,   "wq_varname" : 'ndssi_bnir_oli'         ,'args' : { "NIR_band":'oli06'+s     , "blue_band":'oli02'+s    }},
                #"tm"+s      : {'func': NDSSI_BNIR,   "wq_varname" : 'ndssi_bnir_tm'          ,'args' : { "NIR_band":'tm04'+s      , "blue_band":'tm01'+s    }}
                }

    ti_yu        = {
                    #"msi"     : {'func': TI_yu,        "wq_varname" : 'ti_yu_msi'            ,'args' : {"NIR" : 'msi8a'     , "Red":'msi04'    , "Green":'msi03_agm'}},
                    "oli"+s     : {'func': TI_yu,        "wq_varname" : 'ti_yu_oli'            ,'args' : {"NIR" : 'oli06'+s     , "Red":'oli04'+s    , "Green":'oli03'+s    }},
                    "tm"+s      : {'func': TI_yu,        "wq_varname" : 'ti_yu_tm'             ,'args' : {"NIR" : 'tm04'+s      , "Red":  'tm03'+s     , "Green":'tm02'+s    }}
                    }

    tsm_lym      = {
                    "oli"+s     : {'func': TSM_LYM_OLI,  "wq_varname" : 'tsm_lym_oli'            ,'args' : {"red_band":'oli04'+s    , "green_band":'oli03'+s    }},
                    "msi"+s     : {'func': TSM_LYM_OLI,  "wq_varname" : 'tsm_lym_msi'            ,'args' : {"red_band":'msi04'+s    , "green_band":'msi03'+s    }},
                    "tm"+s      : {'func': TSM_LYM_ETM,  "wq_varname" : 'tsm_lym_tm'             ,'args' : {"red_band":'tm03'+s     , "green_band":'tm02'+s     }}
                    }

    spm_qiu      = {
                    "tm"+s      : {'func': SPM_QIU,  "wq_varname" : 'spm_qiu_tm'             ,'args' : {"red_band":'tm03'+s     , "green_band":'tm02'+s     }},
                    "msi"+s     : {'func': SPM_QIU,  "wq_varname" : 'spm_qiu_msi'            ,'args' : {"red_band":'msi04'+s    , "green_band":'msi03'+s    }}
                    }

    tss_zhang23    = {
                        "msi"+s     : {'func': TSS_Zhang23, "wq_varname" : 'tss_zhang23_msi'     ,'args' : {"G" : 'msi03'+s    , "R":'msi04'+s    , "B":'msi02'+s }},
                        "oli"+s     : {'func': TSS_Zhang23, "wq_varname" : 'tss_zhang23_oli'     ,'args' : {"G" : 'oli03'+s    , "R":'oli04'+s    , "B":'oli02'+s }},
                        "tm"+s      : {'func': TSS_Zhang23, "wq_varname" : 'tss_zhang23_tm'      ,'args' : {"G" : 'tm02'+s     , "R":'tm03'+s     , "B":'tm01'+s  }}
                        }
    tss_GRB       = {
                        "msi"+s     : {'func': TSS_GreenRedBlue, "wq_varname" : 'tss_grb_msi'     ,'args' : {"G" : 'msi03'+s    , "R":'msi04'+s    , "B":'msi02'+s }},
                        "oli"+s     : {'func': TSS_GreenRedBlue, "wq_varname" : 'tss_grb_oli'     ,'args' : {"G" : 'oli03'+s    , "R":'oli04'+s    , "B":'oli02'+s }},
                        "tm"+s      : {'func': TSS_GreenRedBlue, "wq_varname" : 'tss_grb_tm'      ,'args' : {"G" : 'tm02'+s     , "R":'tm03'+s     , "B":'tm01'+s  }}
                        }
    tss_zhang21    = {
                        "msi"+s     : {'func': TSS_Zhang21, "wq_varname" : 'tss_zhang21_msi'     ,'args' : {"G" : 'msi03'+s    , "R":'msi04'+s }},
                        "oli"+s     : {'func': TSS_Zhang21, "wq_varname" : 'tss_zhang21_oli'     ,'args' : {"G" : 'oli03'+s    , "R":'oli04'+s }},
                        "tm"+s      : {'func': TSS_Zhang21, "wq_varname" : 'tss_zhang21_tm'      ,'args' : {"G" : 'tm02'+s     , "R":'tm03'+s  }}
                        }
    tss_GR       = {
                        "msi"+s     : {'func': TSS_GreenRed, "wq_varname" : 'tss_gr_msi'      ,'args' : {"G" : 'msi03'+s    , "R":'msi04'+s }},
                        "oli"+s     : {'func': TSS_GreenRed, "wq_varname" : 'tss_gr_oli'      ,'args' : {"G" : 'oli03'+s    , "R":'oli04'+s }},
                        "tm"+s      : {'func': TSS_GreenRed, "wq_varname" : 'tss_gr_tm'       ,'args' : {"G" : 'tm02'+s     , "R":'tm03'+s  }}
                        }

    # ---- algorithms are grouped into two over-arching dictionaries ---- 
    algorithms_chla = {"ndci_nir_r"   : ndci_nir_r, 
                       "chla_toming"  : chla_toming, 
                       "chla_3bda"  : chla_3bda, 
                       "chla_tebbs"  : chla_tebbs, 
                       "chla_meris2b" : chla_meris2b, 
                       "chla_modis2b" : chla_modis2b}
    algorithms_tsm  = {"ndssi_rg"     : ndssi_rg  , 
                       "ndssi_bnir"   : ndssi_bnir, 
                       "ti_yu"        : ti_yu     ,
                       "tsm_lym"      : tsm_lym   ,
#                       "tss_zhang23"  : tss_zhang23 ,   
#                       "tss_zhang21"  : tss_zhang21 ,
                       "tss_grb"      : tss_GRB ,
                       "tss_gr"       : tss_GR ,
                       "spm_qiu"      : spm_qiu    }
    return(algorithms_chla,algorithms_tsm)
    


# ------------------------------------------------------------------------------------------------------------------------------------
def ChlA_Toming(dataset, band5, band4, band6, verbose=False):
# ---- Function to calculate ndci from input nir and red bands ----
    return  (dataset[band5] - 0.5 * ( dataset[band4] / dataset[band6] ) )
# ------------------------------------------------------------------------
def ChlA_3BDA(dataset, blue_band, green_band, red_band, verbose=False):
    #
    # as described in Byrne et al 2024:  LAQUA: a LAndsat water QUality retrieval tool for east African lakes
    #
    scale = 1 / 10000.    #the reflectances should be less than one foe this function to make sensse
    return  ((dataset[blue_band] * scale )  -  ( dataset[red_band] * scale * dataset[green_band] * scale ) )

# ------------------------------------------------------------------------
def ChlA_Tebbs(dataset, NIR_band, Red_band, verbose=False) :
    # ---- the paramters are optional but for completeness ... ----
    a0 = -135
    a1 = 451  
    return  (a0 + a1 * ( dataset[NIR_band] / dataset[Red_band] ) )


# ------------------------------------------------------------------------
def NDCI_NIR_R(dataset, NIR_band, red_band, verbose=False):
# ---- Function to calculate ndci from input nir and red bands ----
    return  (dataset[NIR_band] - dataset[red_band])/(dataset[NIR_band] + dataset[red_band])


''' MERIS two band:
    meris2b = 25.28 x MERIS_2BM^2 + 14:85 x / MERIS_2BM – 15.18 
    where MERIS_2BM = band9(703.75–713.75 nm) / band7(660–670 nm)
    closest S2 bands are : B5 (703.9) and B4(664) respectively
    may need to divide by pi 

     MERIS2B = 25.28 * ((ds.msi05 / ds.msi04r)*2) + 14.85 * (ds.msi05 / ds.msi04r) - 15.18

     MODIS
     Chla = 190.34 x MODIS_2BM – 32.45
     where: MODIS_2BM = Band15(743–753 nm) / Band13(662–672 nm)
     closest S2 bands are : B6(740) and B4(664)

     MODIS2B = 190.34 * (ds.msi06/ ds.msi04r) - 32.45
'''
# ------------------------------------------------------------------------
def ChlA_MERIS2B(dataset, band_708, band_665, verbose=False):   #matching MSI bands are 5 and 4
# ---- Functions to estimate ChlA using the MERIS and MODIS 2-Band models, using closest bands from MSI (6, 5, 4) or other
    if verbose: print("ChlA_MERIS two-band model")
    X = dataset[band_708] / dataset[band_665]
    return  (25.28 * (X)**2) + 14.85 * (X) - 15.18


# ------------------------------------------------------------------------
def ChlA_MODIS2B(dataset, band_748, band_667,verbose=False):  #matching MSI bands are 6 and 4
    if verbose: print("ChlA_MODIS two-band model")
    X = dataset[band_748] / dataset[band_667]
    return (190.34 * X) - 32.45

# ---- Normalised Difference Suspeneded Sediment Index NDSSI ---
#    These are essentially
#        red-green / red_+green, or,
#        blue-NIR / blue + NIR

# ------------------------------------------------------------------------
def NDSSI_RG(dataset, red_band, green_band,verbose=False):
    if verbose: print("NDSSI_RG")
    return ((dataset[red_band] - dataset[green_band]) / (dataset[red_band] + dataset[green_band]))

# ------------------------------------------------------------------------
def NDSSI_BNIR(dataset, blue_band, NIR_band,verbose=False):
    if verbose: print("NDSSI_BNIR")
    return ((dataset[blue_band] - dataset[NIR_band]) / (dataset[blue_band] + dataset[NIR_band]))


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
def TI_yu(dataset,NIR,Red,Green,scalefactor=0.01,verbose=False):
    # ---- Turbidity index of Yu, X. et al.
    #    An empirical algorithm to seamlessly retrieve the concentration of suspended particulate matter 
    #    from water color across ocean to turbid river mouths. Remote Sens. Environ. 235, 111491 (2019).
    #    Used in screening turbid waters for mapping floating algal blooms
    #    Initially developed with TM
    #    -TI = ((Red − green) − (NIR − Rgreen)) ^ 0.5
    if verbose: print('TI_yu')
    return  scalefactor * ( ( (dataset[Red] - dataset[Green]) - (dataset[NIR] - dataset[Green]) ) ** 0.5 )
#    return  scalefactor * ((dataset[Red] - dataset[Green]) - ((dataset[NIR] - dataset[Green]) * 0.5)) correction!

# ------------------------------------------------------------------------
def TSM_LYM_ETM(dataset, green_band, red_band, scale_factor=0.0001,verbose=False):
    # ---- Lymburner Total Suspended Matter (TSM)
    # Paper: [Lymburner et al. 2016](https://www.sciencedirect.com/science/article/abs/pii/S0034425716301560)
    # Units of mg/L concentration. Variants for ETM and OLT, slight difference in parameters.
    # These models, developed by leo lymburner and arnold dekker, are simple, stable, and produce credible # 
    # results over a range of observations

    if verbose: print("TSM_LYM_ETM")
    return 3983 * ((dataset[green_band] + dataset[red_band]) * scale_factor / 2) ** 1.6246

# ------------------------------------------------------------------------
def TSM_LYM_OLI(dataset, green_band, red_band, scale_factor=0.0001,verbose=False):
        if verbose: print("TSM_LYM_OLI")
        return 3957 * ((dataset[green_band] + dataset[red_band]) * scale_factor / 2) ** 1.6436


# ------------------------------------------------------------------------
    # Qui Function to calculate Suspended Particulate Model value
    # Paper: Zhongfeng Qiu et.al. 2013 - except it's not. 
    # This model seems to discriminate well although the scaling is questionable and it goes below zero due # to the final subtraction.
    # (The final subtraction seems immaterial in the context of our work (overly precise) and I skip it.) 
def SPM_QIU(dataset,green_band,red_band,verbose=False):
    if verbose: print("SPM_QIU")
    return (
        10. ** (
              2.26 * ((dataset[red_band] / dataset[green_band]) ** 3) -
              5.42 * ((dataset[red_band] / dataset[green_band]) ** 2) +
              5.58 * ((dataset[red_band] / dataset[green_band])) -
              0.72
             )
         #- 1.43
         )


# ------------------------------------------------------------------------
def TSS_QUANG8(dataset,red_band,verbose=False):
    # ---- Function to calculate quang8 value ----
    if verbose: print("TSS_QUANG8")
    return 380.32 * (dataset.red_band) * 0.0001 - 1.7826
    # ---- Quang Total Suspended Solids (TSS)
    # Paper: Quang et al. 2017
    # Units of mg/L concentration


# -------------------------------------------------------------------------------------
#   this is a the Zhang21 model without the exponential model fit
def TSS_GreenRed (dataset, G, R, scale_factor=0.0001, verbose=False):
    return( TSS_Zhang21(dataset , G=G , R=R , scale_factor=scale_factor , with_model=False) )

# -------------------------------------------------------------------------------------
def TSS_Zhang21(dataset, G, R, scale_factor=0.0001, with_model=True , verbose=False):

# ---- Model of Zhang et al 2021 Remote sensing: Landsat Image-Based Retrieval and Analysis of Spatiotemporal 
#      Variation of Total Suspended Solid Concentration in Jiaozhou Bay, China
#
#      Developed for Jiaozhou bay, a limited area with TSS values upto 30 (ie relatively low)
#      Fundamentals are: (G + R) / (G/R)   
#                        = G+R  * R/G  
#      Clearly this scales with the data values so dividing by 10000 is a start point for implementation.
#      Sensible values are returned if the data are divided by 2pi
#      Divide by zero is a problem.
#      To avoid zero division in a consistent manner, I introduce an offset of 1 to all values before scaling. 
#      (This offset is within the noise)
#      In fitting the model, the authors introduce an exponent
#
#                        Y = 0.71 * exp(21.31X) where X is the combination above
#      This leads to ridiculous values, making the model useless. 
#   
    Green  = dataset[G].where(~(dataset[G]>0),1) * scale_factor 
    Red    = dataset[R]                          * scale_factor 
    
    if with_model :
            Green = Green / (2 * np.pi)
            Red   = Red   / (2 * np.pi)
            return(0.71 * np.e**(21.31 * (Green + Red) * (Red / Green)))
    else :
        return( (Green + Red) * (Red / Green) )

# -------------------------------------------------------------------------------------
#   this is a the Zhang23 model without the exponential model fit
def TSS_GreenRedBlue (dataset , G , R , B , scale_factor=0.0001 , verbose=False):
    return( TSS_Zhang23(dataset=dataset , G=G , R=R , B=B , scale_factor=scale_factor , with_model=False) )

# -------------------------------------------------------------------------------------
def TSS_Zhang23(dataset, B , G , R , scale_factor=0.0001 , with_model=True , verbose=False):
#    Model of Zhang 2023 
#    Remote sensing monitoring of total suspended solids concentration in Jiaozhou Bay based on multi-source data
#    Ecological Indicators 154 (2023) 110513
#    Developed for Jiaozhou bay in China, low TSS values (less than 20) with cross-reference to MODIS
#    Fundamentals are: X = (G + R) / ( B/R ) = 
#                          (G + R) * ( R/B )
#    This is not scale-free; data must be in reflectance values.
#    Zero values in the blue must be mitigated to avoid nans coming in.     
#    
#    A model is then fitted empirically, viz:
#                          TSS = 1.20 × exp(14.44 * X)
#    The inclusion of an exponent makes values from this function ridiculous, in general
#    HOWEVER: The band combinaton BEFORE the model fit is also useful, showing patterns consistent with NDSSI and Lymburner to which it is closely related
#    
#
    Blue   = dataset[B].where(~(dataset[B]>0),1) * scale_factor 
    Red    = dataset[R]                          * scale_factor 
    Green  = dataset[G]                          * scale_factor 
        
    if with_model :
        Green = Green / (2 * np.pi)
        Red   = Red   / (2 * np.pi)
        Blue  = Blue  / (2 * np.pi)

        return  (1.20 * np.e**( 14.44 * (Green + Red) * (Red / Blue)  )  )
    else:
        return  ( ((Green + Red)*(Red / Blue)) )   


def create_OWT_response_models(path='/home/jovyan/dev/deafrica_water_quality/reference_data/',agm=False):   
    # --- A function to read in Vagelis' reference spectra for Optical Water Types and return 
    #     the expected response (vector) of each sensor to each OWT.
    #     
    #     All water types are included, but inland water types are 1-13
    #     The data are spectral response curves for each OWT, with a 1-nm step
    #     The longest wavelength is 800nm
    #     The response for a specific instrument/band, say oli01, is estimated by integrating the spectrum over the instrument response 
    #     - ideally that is a convolution, using  the instrument response functions, however   
    #     - I settle for an average over the central range
    #     - Checked on 2025-12-12. Corrected the sum to be the mean over the band interval. Corrected the band interval for TM bands in the reference data
    # --- read in the data ---
    
    name = 'Vagelis_OWT_allwaters_mean_standardised.csv'
    suffix = ''
    if agm          : suffix = '_agm'
    # --- read the spectral reflectance models for the optial water types ---
    filename = path+name
    with open(filename,'r') as f:
        OWT_spectra = pd.read_csv(f)
        OWT_spectra = OWT_spectra[OWT_spectra.index < 13]    # --- limit to the inland water types --

    # --- read in the wavelengths for the bands in oli, tm and msi (spectal response models) ---
    data = {}
    for sensor in 'msi','tm','oli':
        name = 'sensor bands-'+sensor+'.csv'
        filename = path+name
        with open(filename,'r') as f:
             sensor_data = pd.read_csv(f)
             # rename columns to avoid clashes with reserved words
             sensor_data.rename(columns={'max': 'l_max','min': 'l_min'}, inplace=True)
    
        a = sensor_data
        # list the bands  that are going to be relevant , ie., less than 800nm
        band_list = (a[a['l_max']<800].band_name).values # & set(ds.data_vars) 
        # --- set up a  dataframe to take results for this instrument ---
        labels = [] 
        for i in np.arange(1,14): labels = np.append(labels,'OWT-'+str(i))

        inst_OWT = pd.DataFrame(
            data = {'OWT' : np.arange(1,14,1)}, 
            index = labels
            )
        # --- run through the bands ...
        for band_name in band_list :
            # determine the integration interval based on the central wavlength and the width; extract these as integers
            delta =      a[a['band_name']==band_name]['width'].values * 0.8 *0.5
            start = int((a[a['band_name']==band_name]['central'].values - delta)[0]) 
            end   = int((a[a['band_name']==band_name]['central'].values + delta)[0]) 
            #print(sensor,band_name, start,end)

            # find the start and end column numbers in the response functions and sum over those for each OWT
            # must allow for the spectral band width  - average not the sum
            b = OWT_spectra.loc[:,str(start):str(end)].T.mean()    
            # add to the data frame
            inst_OWT.insert(inst_OWT.columns.size,band_name+suffix,(b.values))
        # --- add this data frame to the data dictionary ---
        data[sensor] = inst_OWT
        # --- write to a csv file in the current location --- 
        inst_OWT.to_csv(sensor+'_OWT_vectors.csv')
    return(data)
    
def OWT (ds,instrument,OWT_vectors,agm=False,dp_corrected = False, verbose=True,test=False):
    # --- this version uses a long-hand approch to calculating the dot product. Inelegant but simple.
    # --- identify the instrument bands that are relevant and in the dataset
    resample_rate=1  #legacy stuff
    if verbose or test : print('\n calculating optical water type (OWT), sensor = ',instrument,' ....\n')
    agm          = False
    dp_corrected = False
    suffix=''
    if agm          : suffix = suffix + '_agm'
    if dp_corrected : suffix = suffix + 'r'
    for col in OWT_vectors.columns.values:     #--- this loop renames the columns so that we can match them with the data variables
        if col.find(instrument)>-1:
            OWT_vectors = OWT_vectors.rename(columns={col:col+suffix})  
    band_list = list(set(ds.data_vars) & set(OWT_vectors.columns.values))
    band_list.sort()
    # --- ditch unnecessary columns and rows in the vectors table and calculate the magnitude of each vector
    OWT_vectors           =  OWT_vectors.drop(columns=list(set(OWT_vectors.columns.values) - set(band_list)))
    OWT_vectors['length'] = ((OWT_vectors**2).sum(axis=1))**0.5  # calculate the length of each vector  
    OWT_vectors = OWT_vectors.reset_index().rename(columns={'index':'OWT'})    # brings the OWT labels in as a normal column,'OWT'

    #--- loop through the Optical water types ---
    start     = True
    for OWT in OWT_vectors['OWT']:
        vec = OWT_vectors[OWT_vectors['OWT'] == OWT]  # the vector for this OWT
        OWT_index = int(OWT[OWT.find('-')+1:])  # the OWT numnber
        if test: print('OWT number:',OWT_index)
        #print(OWT_vectors[OWT_vectors['OWT']==OWT])
        # --- create a dataset for this OWT based on one of he instrument bands ---
        varname   = band_list[0]
        if start:
            start     = False
            # working variiables
            mydataset = xr.Dataset({'owt_current' : ds[varname]})
#            mydataset = xr.Dataset({'owt_current' : ds[varname][:,::resample_rate,::resample_rate]})
            mydataset['owt_cos_max']        = mydataset['owt_current']*0 + -2  #a number smaller than any cosine
            mydataset['owt_cos'    ]        = mydataset['owt_current']*0 
            mydataset['owt_closest']        = mydataset['owt_current']*0 + OWT_index
            mydataset['self_product']       = mydataset['owt_current']*0 
            mydataset['vector_product']     = mydataset['owt_current']*0 
            if test: print('prevailing owt is :',mydataset.owt_closest.median())        


        mydataset['self_product'  ] = 0                      
        mydataset['vector_product'] = 0                      
        for band in band_list:
            if test: print('processing band: ',band)
            mydataset['self_product'  ]  = mydataset['self_product'  ] + ds[band]**2
            mydataset['vector_product']  = mydataset['vector_product'] + ds[band]*vec[band].item()
            
        mydataset['self_product'] = mydataset['self_product'  ] ** 0.5
        mydataset['owt_cos'     ] = mydataset['vector_product'] \
                     / (mydataset['self_product'] * vec['length'].values)
        mydataset['owt_closest'] = xr.where(mydataset['owt_cos'] > mydataset['owt_cos_max'] , OWT_index ,            mydataset['owt_closest'])
        mydataset['owt_cos_max'] = xr.where(mydataset['owt_cos'] > mydataset['owt_cos_max'] , mydataset['owt_cos'] , mydataset['owt_cos_max'])
        if test: print('prevailing owt is :',mydataset.owt_closest.median())        
        
    
    if verbose or test : print('....done')
    if verbose         : 
        data =  [
                [0,'zero - unresolved (error)'],
                [3,'oligotrophic (clear)'],
                [9,'oligotrophic (clear)'],
                [13,'oligotrophic (clear)'],
                [1,'eutrophic and blue-green'],
                [2,'eutrophic and blue-green'],
                [4,'eutrophic and blue-green'],
                [5,'eutrophic and blue-green'],
                [11,'eutrophic and blue-green'],
                [12,'eutrophic and blue-green'],
                [6,'hyper-eutrophic and green-brown'],
                [7,'hyper-eutrophic and green-brown'],
                [8,'hyper-eutrophic and green-brown'],
                [10,'hyper-eutrophic and green-brown']
                ]
        data.sort()
        columns = ['OWT','description']
        
        df   = pd.DataFrame(data=data,columns=columns)
        owt  = mydataset.owt_closest.median().item()
        if np.isnan(owt) : desc = 'nan value returned'
        else : desc = df[df['OWT']==owt]['description'].item()
        print('Prevailng water type is ',owt,' :  ',desc)
    return(mydataset['owt_closest'])

def OWT_pixel_deprecated (ds,instrument,water_frequency_threshold=0.8,resample_rate=3,verbose=True, test=True):
    # --- Determine the Open Water Type for each pixel, over areas that are usually water. 
    # --- 'instrument' is a dictionary established while building the dataset
    # --- 'resample_rate' is the spatial resample step to reduce the memory required
    # --- The returned lattice is on the same t,x,y coordinates as the original, after resampling back to it
    # --- this code will need revisiting to support non-geomedian data which has more bands available. 
    # --- Also, right now, msi and oli are dealt with separately rather than as alternatives ... and it would be nice to remove this 'hard-wiring'.
    # --- Memory intensive due to the use of vector multiplication (dot products). 
    #      A less elegant coding approch might be more memory efficient ----
    # 
    if verbose: print('Determining the Optical Water Type ...')
    
    # estimated spectra for each optical water type for each sensor (MSI, OLI, TM) calculated from full spectra table provided by Vagelis Spyrakos
    #columns are bands, rows are OWT
    #1	2	3	4	5	6	7
    
    owt_data_msi = np.asarray ([0.000518281,0.002873764,0.003495403,0.001515508,0.002204739,0.003842756,0.005026098,
                    0.001310923,0.007407241,0.007373993,0.003239791,0.001309939,0.000351881,0.00044227,               
        0.002720793,0.012390417,0.00737146,0.001458816,0.000422552,0.000142766,0.00018326,
        0.001011837,0.006135417,0.006416972,0.00401114,0.001825363,0.000454228,0.000554565,
        0.001368776,0.005729538,0.004259349,0.00409354,0.001965091,0.001276867,0.001526747,
        0.000947881,0.005756157,0.006822492,0.003394953,0.002302141,0.000694368,0.000822148,
        0.000720924,0.003734303,0.004729343,0.002251117,0.003381287,0.002206305,0.002708644,
        0.000830314,0.004805815,0.005933108,0.003108917,0.002882319,0.001181092,0.001454993,
        0.001877709,0.009255961,0.007552662,0.002441745,0.000847245,0.000279124,0.000375539,
        0.000842475,0.00240552,0.002787709,0.005383652,0.003098927,0.001542782,0.002194582,
        0.000746178,0.004633521,0.005087786,0.00500395,0.002445958,0.000762379,0.000962499,
        0.001439293,0.006599122,0.005582806,0.003590672,0.001796926,0.000768249,0.000934859,
        0.006322908,0.014258851,0.002314198,0.000275429,8.56521E-05,6.41172E-05,0.000110569]).reshape(13,7)

    #	oli bands:1,2,3,4, rows are owt:
    owt_data_oli = np.asarray ([0.000536567,0.000860402,0.002160328,0.001178919,
         0.001346095,0.002267667,0.004757344,0.002459032,
         0.002818286,0.0041035,0.004739951,0.001088903,
         0.001034619,0.001850067,0.004198279,0.003021194,
         0.001429095,0.001886083,0.002822574,0.00284571,
         0.000971,0.001697917,0.004364393,0.002620484,
         0.000754857,0.001114315,0.002976344,0.001750548,
         0.000856452,0.00142315,0.003775541,0.002414677,
         0.001945476,0.002931367,0.004860984,0.001833516,
         0.000936538,0.000772627,0.001869377,0.003541968,
         0.000764871,0.001385817,0.00335777,0.003580806,
         0.001499333,0.002115267,0.003644426,0.00264,
         0.006800333,0.00588125,0.001539375,0.000199513]).reshape(13,4)

    # ---- make an empty dataset for the OWT reference data
    # ---- owt types are stored in their own dimension to support vector multiplication
    owt_list = ['owt1','owt2','owt3','owt4','owt5','owt6','owt7','owt8','owt9','owt10','owt11','owt12','owt13']
    owt_msi = xr.DataArray(owt_data_msi,
                           dims=("owt","band"),
                           coords={"owt": owt_list, "band": np.arange(1,8,1)},
                           attrs = {"desc" : "optical water types - characteristic reflectances for msi"})
    owt_oli = xr.DataArray(owt_data_oli,
                    dims=("owt","band"),
                    coords={"owt": owt_list, "band":np.arange(1,5,1)},
                    attrs = {"desc" : "optical water types - characteristic reflectances for oli"})
    OWT = {'msi_agm': owt_msi,'oli_agm':owt_oli}
    shortname = instrument[0:3]  # --- string of the prefix of the variable name, used later ---

    # ---- build a working dataset with the msi and oli values stored in their own dimension ----
    bandlists = {"msi_agm" : {"bands" : ['msi02','msi03','msi04','msi05','msi06','msi07'],
                              "suffix": '_agmr',"dim_name": 'msi_band',
                              "var_name":'msi_vals', 
                              "band_index": [1,7]},
                 "oli_agm" : {"bands" : ['oli02','oli03','oli04'                        ],
                              "suffix": '_agmr',"dim_name": 'oli_band',
                              "var_name":'oli_vals', 
                              "band_index": [1,4]}}    
    bandlist = bandlists[instrument]['bands'];    suffix = bandlists[instrument]['suffix'];     dim_name = bandlists[instrument]['dim_name']; 
    var_name = bandlists[instrument]['var_name']; i,j    = bandlists[instrument]['band_index'] 
    owt_data = np.asarray(OWT[instrument])
    
    # --- make a new resampled (downsized) dataset ---
    mydataset = xr.Dataset({'watermask' : ds['watermask'][:,::resample_rate,::resample_rate]})
    for name in bandlist: 
        mydataset[str(name)+suffix] = ds[str(name)+suffix]

    # ---- the following looked promising but will only work on an array, not a dataset:
    # mydataset = ds[varlist][:,::resample_rate,::resample_rate].copy()  #makes an explicit copy in memory

    # ---- create a dimension for the surface reflectance data. This longhand code is because I can't see how to easily iterate through the band list.
    if instrument == 'msi_agm' :  data_stack = np.dstack([mydataset['msi02_agmr'],
                                                          mydataset['msi03_agmr'],
                                                          mydataset['msi04_agmr'],
                                                          mydataset['msi05_agmr'],
                                                          mydataset['msi06_agmr'],
                                                          mydataset['msi07_agmr']])
    if instrument == 'oli_agm' :  data_stack = np.dstack([mydataset['oli02_agmr'],
                                                          mydataset['oli03_agmr'],
                                                          mydataset['oli04_agmr']])
    
    mydataset = mydataset.assign_coords({dim_name: bandlist})
    mydataset[var_name] = (('time', 'y', dim_name,'x'), data_stack.reshape(mydataset.time.size,mydataset.y.size,np.size(bandlist),mydataset.x.size))
    mydataset = mydataset.transpose('time','y','x',dim_name,...)  #to allow the matrix multiplication the bands dimension needs to be transposed to the end

    # ---- we multiply each pixel vector by the full OWT matrix, adding yet another dimension to the dataset (dimensions are now: time,x,y,band_name,owt_number) 
 
    # --- bring in the relevant parts of the  msi_owt matrix, and shape to shape of 6 (bands) by 13 (owts) 
    # ---- add a dimension for the OWT type
    owt_dim_values = list(np.array(OWT[instrument]['owt']))   # --- in practice same for any instrument
    mydataset      = mydataset.assign_coords(owt_num=owt_dim_values)

    # ---- calculate the dot product between each pixel vector and each owt type vector, i and j are used to skip band1 since we dont have it ----
    pixel_x_owt_data = np.array(mydataset[var_name]) @ owt_data[:,i:j].T   
   
    # ---- add results back into the dataset ----
    mydataset[shortname+'_x_owt'] = (('time', 'y', 'x','owt_num'), pixel_x_owt_data) 

    # ---- mydataset now has, for every pixel, the dot product with the spectral reference vector!!
    # ---- but we need the cosine value...so scale it to the range of -1 to +1 
    
    # ---- calculte the self product (the scale) of each of hte OWT reference vectors:
    owt_scale     = np.sum((owt_data[:,i:j] * owt_data[:,i:j]).T,axis=0)**0.5

    # ---- calculate the scale of each pixel vector 
    mydataset[shortname+'_scale']  = ((mydataset[shortname+'_vals'] * mydataset[shortname+'_vals']).sum(dim=(dim_name))) ** 0.5
    
    # ---- now to the crunch...
    mydataset[shortname+'_owt_cosine'] = \
        (mydataset[shortname+'_x_owt'] / mydataset[shortname+'_scale']) / owt_scale

    # ---- now find the owt closest to the msi vector (the largest cosine)
    # ---- to avoid nan problems use np.argmax, then the array is  brought back into the dataset
    mydataset[shortname+'_owt'] = (('time','y','x'),np.argmax(np.array(mydataset[shortname+'_owt_cosine']),axis=3)+1) 

    # ---- replace zeros (where the scale of the pixel vector is zero) with nodata
    mydataset[shortname+'_owt']= xr.where(mydataset[shortname+'_scale']>0,mydataset[shortname+'_owt'],np.nan)

    # --- interpolate back to original grid 
    # --- up-sample the data array up to the original grid, and mess about to deal with the fact that interp doesn't do what it should on the edges!

    # --- replace nans with a fill value ---
    da =  mydataset[shortname+'_owt']
    fill_value = 14
    da   = xr.where(np.isnan(da),fill_value,da)

    # --- interpolate to the original coordinates, and fill out any gaps including for years prior to the msi sensor being available ---
    da   = da.interp(coords={'time' : ds.time,'x' : ds.x,'y' : ds.y},method='nearest',kwargs={'bounds_error': False,'fill_value':fill_value})
    # ---- reduce coverage to water areas ---
    da   = xr.where(~np.isnan(ds.watermask),da,np.nan)
    # ---- replace the fill values with median values for the pixel, or year
    glob_med   = da.where(da!=fill_value).median()
    pixel_med  = da.where(da!=fill_value).median(dim='time')
    annual_med = da.where(da!=fill_value).median(dim=('x','y'))
    da   = xr.where(da==fill_value, pixel_med ,da)
    da   = xr.where(da==fill_value, annual_med,da)
    
    # ---- and we are done!  Note that I have code elsewhere to calculate the OWT of a median set of pixels, rather than every pixel ---
    
    # ---- return the (decimated) pixel-level OWT types
    gc.collect()  #---not sure if this helps but worth a try---
    return da

def hue_adjust_parameters():
    # --- a function to set the hue adjustment parameters; a quintic polynomial model ---
    return(pd.DataFrame(
            data = {
                'label' :  ['Resolution','a5','a4' ,'a3' , 'a2',  'a', 'offset'],
                'msi' :  [20, -161.23, 1117.08, -2950.14, 3612.17, -1943.57, 364.28],
                'oli' :  [ 30, -52.16, 373.81,  -981.83, 1134.19, -533.61, 76.72],
                'tm'  :  [ 30, -84.94, 594.17, -1559.86, 1852.50, -918.11, 151.49]
            }))

def hue_adjust(dataset,instrument='msi') :  # revised version a bit more compact and tidy 
    # hue adjustment coefficients for MSI. This is the final step in calculating the hue value
    # I am sure that there are more efficient ways to code this as a matrix multiplication but at least this is transparent! 
    # ---- this function makes an adjustment to the hue to produce the final value ----
    #      These quintic functions run off the scale if the hue value is less than about 25 or greater than about 240.
    #      It may be necessary to put a condition on the adjustment to avoid invalid results.
    #      However, provided only water pixels are included, results seem okay
    
    if instrument in ('msi_agm','oli_agm','tm_agm'): instrument = instrument[0:instrument.find('_agm')]
    hap          = hue_adjust_parameters()
    coefficients = hap[instrument][hap[np.isin(hap.label,['a5','a4','a3','a2','a','offset',])].index].values   
    
    dataset['hue_delta'] = (dataset['hue']/100)**5 * coefficients[0]  +   \
               (dataset['hue']/100)**4 *coefficients[1] + \
               (dataset['hue']/100)**3 *coefficients[2] + \
               (dataset['hue']/100)**2 *coefficients[3] + \
               (dataset['hue']/100)**1 *coefficients[4] + \
               (dataset['hue']/100)**0 *coefficients[5]
    dataset['hue'] = dataset['hue'] + dataset['hue_delta']
    return(dataset.drop_vars('hue_delta'))
    
def hue_adjust_old_version(dataset) :
    # hue adjustment coefficients for MSI. This is the final step in calculating the hue value
    # I am sure that there are more efficient ways to code this as a matrix multiplication but at least this is transparent! 
    # ---- this function makes an adjustment to the hue to produce the final value ----
    deltahuemsi = (-161.23  , 1117.08 , -2950.14  , 3612.17  , -1943.57  , 364.28)
    deltahueoli = ( -52.16  , 373.81  ,  -981.83  , 1134.19  ,  -533.61  , 76.72)
    deltahueetm = ( -84.94  , 594.17  , -1559.86  , 1852.50  ,  -918.11  , 151.49)
    dataset['hue_delta'] = (dataset['hue']/100)**5 * deltahuemsi[0]  +   \
               (dataset['hue']/100)**4 *deltahuemsi[1] + \
               (dataset['hue']/100)**3 *deltahuemsi[2] + \
               (dataset['hue']/100)**2 *deltahuemsi[3] + \
               (dataset['hue']/100)**1 *deltahuemsi[4] + \
               (dataset['hue']/100)**0 *deltahuemsi[5]
    dataset['hue'] = dataset['hue'] + dataset['hue_delta']
    dataset = dataset.drop_vars('hue_delta')
    return(dataset)

def chromatic_coefficient_parameters():
    msi = pd.DataFrame({
        'nm'   : ['R400','R490'  ,'R560'  ,'R665'   ,'R705' ,'R710'],
        'band' : [''    , '2'    , '3'    , '4'     , '5'   ,''],
        'name' : [''    , 'msi02', 'msi03', 'msi04' , 'msi05'   ,''],
        'X'    : [ 8.356 , 12.040 , 53.696 , 32.028 , 0.529 , 0.016 ],
        'Y'    : [ 0.993 , 23.122 , 65.702 , 16.808 , 0.192 , 0.006 ],
        'Z'    : [ 43.487, 61.055 , 1.778  , 0.015  , 0.000 , 0.000 ],
    }, )

    oli = pd.DataFrame({
        'nm'   : ['R400' , 'R443' , 'R482' , 'R561' , 'R655' , 'R710'],
        'band' : [  ''   ,    '1' ,   '2'  ,   '3'  ,   '4'  ,   ''  ],  
        'name' : [  ''   , 'oli01' ,'oli02' ,'oli03', 'oli04',   ''  ],  
        'X'    : [ 2.217 , 11.053 ,  6.950 , 51.135 , 34.457 , 0.852 ],
        'Y'    : [ 0.082 , 1.320  , 21.053 , 66.023 , 18.034 , 0.311 ],
        'Z'    : [ 10.745 , 58.038 , 34.931 , 2.606 ,  0.016 , 0.000 ]
        })

    tm = pd.DataFrame({
        'nm'   : [ 'R400' ,  'R485', 'R565' , 'R660' , 'R710' ],
        'band' : [  ''    ,    '1' ,  '2'   ,   '3'  , ''     ],
        'name' : [  ''    ,  'tm01', 'tm02' ,  'tm03', ''     ],
        'X'    : [ 7.8195 , 13.104 , 53.791 , 31.304 , 0.6463 ],
        'Y'    : [ 0.807  , 24.097 , 65.801 , 15.883 , 0.235  ],
        'Z'    : [ 40.336 , 63.845 , 2.142  , 0.013  , 0.000  ],
        })

    # --- put the parameters into a dictionary
    return({'msi':msi,'oli':oli,'tm':tm})


def hue_calculation(dataset,instrument='',test=False,verbose=False) : 
    #---- Hue is calculated by conversion of the wavelengths to chromatic coordinates using sensor-specific coefficients
    #- Method is as per Van Der Woerd 2018.
    #- More accurate hue angles are retrieved if more bands are used - but the visible bands are most important
    #- results for ETM+ are therefore  less accurate than for MSI and OLI?
    #- the OLI geomedian lacks band 1, so it cannot be used. This leaves a gap in the data.
    #- examiation of a time series shows clear patterns. Oli data give lower values than msi and tm, which are in good agreement 

    #enter the x,y,z msi chromatic coefficients...
     #---- the hue is calculated by conversion of the wavelengths to chromatic coordinates using sensor-specific coefficients
    instr = instrument
    if instr in ('msi_agm','oli_agm','tm_agm'):  
        instr = instr[0:instr.find('_agm')]  
        agm = True
    else: 
        agm = False
    
    ccs = chromatic_coefficient_parameters()[instr]

    # determining the available bands gets a bit messy due to continengcies...
    required_bands = ccs['name'][ccs['name']!=''].values         # pos

    # --- make two lists of band names ---
    dsbands = []
    for name in required_bands:
        if agm                     : name = name + '_agm'
        if name in dataset.data_vars    : dsbands.append(name)

    Cdata         = xr.zeros_like(dataset).drop_vars(dataset.data_vars)
    Cdata['hue'] = ('time','y','x'), np.zeros((dataset.sizes['time'],dataset.sizes['y'],dataset.sizes['x']))

    if np.size(required_bands) != np.size(dsbands) : 
        print('\n Aborting hue calculation for instrument ',instr,' due to lack of necessary data bands\n')
        Cdata['hue'] = ('time','y','x'), np.zeros((dataset.sizes['time'],dataset.sizes['y'],dataset.sizes['x']))*np.nan
        return(Cdata['hue'])
    
    for XYZ in 'X','Y','Z':
        Cdata[XYZ] =  ('time','y','x'), np.zeros((dataset.sizes['time'],dataset.sizes['y'],dataset.sizes['x']))
        for band in required_bands:
            dsband       = dsbands[list(required_bands).index(band)] 
            coeff        = ccs[ccs.name==band][XYZ].values
            Cdata[XYZ]   = dataset[dsband] * coeff + Cdata[XYZ]

    Cdata["XYZ"] = Cdata['X'] + Cdata['Y'] +  Cdata['Z']
    Xwhite =  Ywhite = 1 / 3.
    
    # ---- normalise the X and Y parameters and conver to a delta from white:
    Cdata["X"] = Cdata['X'] / Cdata['XYZ'] - Xwhite 
    Cdata["Y"] = Cdata['Y'] / Cdata['XYZ'] - Ywhite
    Cdata["Z"] = Cdata['Z'] / Cdata['XYZ']
    # ---- convert vector to angle ----
    Cdata['hue'] = np.mod(np.arctan2(Cdata['Y'],Cdata['X'])*(180.00/np.pi) +360.,360)  
    Cdata = hue_adjust(Cdata,instr)
    
    # ---- this gives the correct mathematical angle, ie. from 0 (=east), counter-clockwise as a positive number
    # ---- note the 'arctan2' function, and that x and y are switched compared to expectations
    
    return(Cdata.hue)
    
def geomedian_hue (ds_annual,water_mask,verbose=False,test=False):
    # --- a function to calculate the hue value of the geomedian, allowing for the possibility of multiple sensors at each time point
    # (band one is missing from the oli agm, but perhaps we will be able to do without it)
    # To combine the hue from multiple sensors we take a weighted mean. 
    count = 0   
    for inst in list(('msi','oli','tm')):
        inst_agm = inst+'_agm'
        if inst_agm in ds_annual.data_vars: 

            # --- calculate the HUE for this sensor
            hue_data = hue_calculation(ds_annual,inst_agm,test=test,verbose=verbose)
            
            # --- set nans to zero, and also in the agm_count variable
            ds_annual[inst_agm+'_count'] = xr.where(~np.isnan(ds_annual[inst_agm+'_count']),ds_annual[inst_agm+'_count'],0)
            #negate the counts where there is no data produced 
            ds_annual[inst_agm+'_count'] = xr.where(~np.isnan(hue_data)                    ,ds_annual[inst_agm+'_count'],0)
            hue_data                     = np.where(~np.isnan(hue_data)                    ,hue_data                    ,0)
                      
            if count == 0: 
                mean_hue  =  hue_data * ds_annual[inst_agm+'_count']
                agm_count =             ds_annual[inst_agm+'_count']
            else :    
                mean_hue  = mean_hue  + hue_data * ds_annual[inst_agm+'_count']
                agm_count = agm_count + ds_annual[inst_agm+'_count']
            count = count + 1
         
            #hue_data = np.where(~np.isnan(water_mask),hue_data,np.nan)   # new code with mask 

            # --- this retains a value for each instument, not essential but useful during developement
            ds_annual[inst_agm+'_hue'] = ('time','y','x'),hue_data
            ds_annual[inst_agm+'_hue'] = xr.where(water_mask,ds_annual[inst_agm+'_hue'],np.nan)

            #ds_annual[inst_agm+'_hue'] = xr.where(water_mask ,ds_annual[inst_agm+'_hue'] , hue_data)

    # --- divide by the total count to get the actual mean, then trim back to relevant values and areas
    mean_hue = mean_hue / agm_count
    ds_annual['agm_hue'] = (ds_annual.wofs_ann_freq.dims), mean_hue.data
    ds_annual['agm_hue'] = xr.where(water_mask,ds_annual['agm_hue'],np.nan)
    # ---- trim extreme values that can arise 
    ds_annual['agm_hue'] = xr.where(ds_annual['agm_hue']>25,
                                    xr.where(ds_annual['agm_hue'] < 100,ds_annual['agm_hue'],
                                             np.nan),np.nan)
    return(ds_annual)

    
def hue_calculation_old_version(dataset,instrument='msi_agm',test=False,verbose=False) : 
    #---- the hue is calculated by conversion of the wavelengths to chromatic coordinates using sensor-specific coefficients
    ### Colour space transformation on MSI data.
    #- Method is as per Van Der Woerd 2018.
    #- More accurate hue angles are retrieved if more bands are used;
    #- results for ETM+ are therefore  less accurate than for MSI and OLI
    #- Cannot run for OLI at this point, because we don't have band1 in the geomedian
    #- Results should in general be more accurate with more bands, but MSI, OLI and ETM are limited

    #enter the x,y,z msi chromatic coefficients...
    chrom_coeffs = {
        'X': {'msi01':8.356, 'msi02':12.040,'msi03': 53.696,'msi04':32.028,'msi05': 0.529}, #x msi chromaticity
        'Y': {'msi01':0.993, 'msi02':23.122,'msi03': 65.702,'msi04':16.808,'msi05': 0.192},
        'Z': {'msi01':43.487,'msi02':61.055,'msi03':  1.778,'msi04': 0.015,'msi05': 0.000},
        }
    #use all the bands that are in the dataset (band1 is probably missing..):
    if instrument == 'msi_agm' : band_list = ['msi01_agmr','msi02_agmr','msi03_agmr','msi04_agmr','msi05_agmr']
    if instrument == 'msi'     : band_list = ['msi01r'    ,'msi02r'    ,'msi03r'    ,'msi04r'    ,'msi05r']
    
    var_list = []
    for name in band_list:
        if name in dataset.data_vars:
            var_list = np.append(var_list,name)

    #initiate two Datasets with no variables:
    Cdata         = xr.zeros_like(dataset).drop_vars(dataset.data_vars)
    Cdata_summary = dataset.drop_dims(['x','y'])  #summary is not required in the per-pixel processing
    
    n = 1 ; s = np.array([], dtype=np.int8)
    for d in Cdata.dims:        n = n * Cdata[d].size;         s = np.append(s,Cdata[d].size)

    for XYZ in chrom_coeffs.keys() :
        Cdata[XYZ] = Cdata.dims, np.zeros(n).reshape(s)                #-- initiate a dataset
        for var in var_list:
            var_shortname = var[0:5]   #-- drops the 'r', or '_agmr' from the variable name
            Cdata[XYZ] = Cdata[XYZ] + dataset[var] * chrom_coeffs[XYZ][var_shortname]
            
    # ---- normalise the X and Y parameters
    Cdata["Xn"] = Cdata['X'] / ( Cdata['X'] +  Cdata['Y'] +  Cdata['Z'])
    Cdata["Yn"] = Cdata['Y'] / ( Cdata['X'] +  Cdata['Y'] +  Cdata['Z'])
    Xwhite =  Ywhite = 1 / 3.
    # ---- calculate the delta to white ----
    Cdata['Xnd'] = Cdata['Xn'] - Xwhite; 
    Cdata['Ynd'] = Cdata['Yn'] - Ywhite; 

    # ---- convert vector to angle ----
    Cdata['hue'] = np.mod(np.arctan2(Cdata['Ynd'],Cdata['Xnd'])*(180.00/np.pi) +360.,360)  

    # ---- this gives the correct mathematical angle, ie. from 0 (=east), counter-clockwise as a positive number
    # ---- note the 'arctan2' function, and that x and y are switched compared to expectations

    # ---- code below is not used for pixel level processing, but is used by others / later!
    Cdata_summary['Xnd'] = Cdata['Xnd'].where(dataset['wofs_ann_freq']>0.9).median(dim=('x','y'))
    Cdata_summary['Ynd'] = Cdata['Ynd'].where(dataset['wofs_ann_freq']>0.9).median(dim=('x','y'))  
    Cdata_summary['hue'] = np.mod(np.arctan2(Cdata_summary['Ynd'],Cdata_summary['Xnd'])*(180.00/np.pi) +360.,360)  

    #apply the hue adjustment - only do it once!
    if test: print('Average Hue values pre-adjustment :', Cdata_summary['hue'].values.round(1)) 
    Cdata         = hue_adjust(Cdata)
    Cdata_summary = hue_adjust(Cdata_summary) 
    if test: print('Average Hue values post-ajustment :', Cdata_summary['hue'].values.round(1)) 

    #plot the vectors - dev code only
    if verbose:
        print('Average Hue values post-ajustment :', Cdata_summary['hue'].values.round(1)) 
        if test:
            print('Hue vectors :', Cdata['Ynd'].mean(dim=('x','y')).values.round(4),Cdata['Ynd'].median(dim=('x','y')).values.round(4) )
            print('Hue vectors :', Cdata['Xnd'].mean(dim=('x','y')).values.round(4),Cdata['Xnd'].median(dim=('x','y')).values.round(4) )
        if test:Cdata['Ynd'].sel(time=slice('2019','2020')).plot(col='time',robust=True,vmin=0)#,vmax=360,cmap='hsv')
    return Cdata['hue'],Cdata_summary['hue']  #the second output is not required for pixel-level processing, but is used by some of my code. 

# --- Dark pixel correction - preparing inputs 
#    'dp_adjust_parameters' dictionary controls which variables are used as a reference, 
#     and which are changed, in the dark-pixel correction.

#     Revised code 2025-10-07 to cater for non-geomedian situations. 

def apply_R_correction(ds,
                       instr_list,
                       water_mask, drop = True,test=False,verbose=False) :
        
    dp_adjust_parameters = { 
        'msi': {'ref_var':'msi12' , 'var_list': ['msi04','msi03','msi02','msi01','msi05','msi06', 'msi07']},
        'oli': {'ref_var':'oli07' , 'var_list': ['oli04','oli03','oli02','oli01']},
        'tm' : {'ref_var': 'tm07' , 'var_list': [ 'tm04', 'tm03', 'tm02', 'tm01']}
        }
    # --- check the instrument list against the dataset  --- 
    instr_list = list(set(instr_list) & set(ds.data_vars))    
    # ---- check the reference variables against the dataset --- 
    templist  = instr_list
    for instr in templist:
        # item_number = templist == instr #  # --- this seemed like a mistake... does not always work... 
        item_number = np.where(np.isin(templist,instr))[0].item()  
        agm        = False ; suffix = ''
        if instr.find('_agm') > 0:
            agm     = True; suffix = instr[instr.find('_agm'):]
            instr   = instr[0:instr.find('_agm')]
        if not    dp_adjust_parameters[instr]['ref_var']+suffix in ds.data_vars:
            instr_list.pop(item_number)    

    ds = R_correction(ds=ds,
                      instr_list=instr_list,
                      dp_adjust_parameters=dp_adjust_parameters,
                      water_mask = water_mask,
                      drop = drop,
                      verbose=verbose,
                      test=test)
    return(ds)


def R_correction(ds,instr_list,dp_adjust_parameters,water_mask,drop=True,verbose=False,test=True):
    #--- Rayleigh correction - dark pixel adjustment ---
    #--- for each variable in the list, reduce by the value of the ref_variable, over target areas ---
    #--- target areas are areas withing the provided water mask 
    #-------------------------------------------------------------------------------------------------------------
    # --- the 'dp_adjust' dictionary passed in as an argument controls which variables are used as a reference, and which are changed 
    # --- it is assumed at this point that the relvant variables are in the dataset.  (Checks are done in the calling function).
    
    for instr in instr_list:
        agm = False;  suffix = ''
        if  instr.find('_agm') > 0 :    
            suffix = instr[instr.find('_agm'):]
            agm = True; 
            instr = instr[0:instr.find('_agm')]
       
        reference_var = dp_adjust_parameters[instr]['ref_var']+suffix
        for target_var in dp_adjust_parameters[instr]['var_list']:
            target_var = target_var+suffix
            new_var    = target_var+'r'
            if new_var in ds.data_vars: ds=ds.drop_vars(new_var)
            if not target_var in ds.data_vars :
                print('---variable -->  ',target_var,'  <-- anticipated but not found in the dataset (non-fatal)')
            else:
                ds[new_var] = ds[reference_var]*0.0
                ds[new_var] = xr.where(ds[target_var]>0,xr.where(ds[target_var] > ds[reference_var] ,ds[target_var] - ds[reference_var] , ds[target_var]),np.nan)
                
                ds[new_var] = xr.where(water_mask,ds[new_var],ds[target_var]) 
                #print('stopping')
                #return(ds)
                if drop :
                    ds = ds.drop_vars(target_var)
                    ds = ds.rename   ({new_var : target_var})

    if test or verbose and not drop : #plot graphs illustrating the shift in the cumulative distribution
        
            quantiles = np.arange(0,1,.01)
            plt.plot(ds['msi02_agmr'].quantile(quantiles),quantiles,"b-")
            plt.plot(ds['msi02_agm'].quantile(quantiles),quantiles,"b--")
            plt.plot(ds['msi03_agmr'].quantile(quantiles),quantiles,"g-")
            plt.plot(ds['msi03_agm'].quantile(quantiles),quantiles,"g--")
            plt.plot(ds['msi04_agmr'].quantile(quantiles),quantiles,"r-")
            plt.plot(ds['msi04_agm'].quantile(quantiles),quantiles,"r--")
   
    if verbose : print('R_correction completed normally')
    return(ds)
    
def R_correction_old(ds,dp_adjust,instruments,water_frequency_threshold=0.9,verbose=False,test=True):
    #--- Rayleigh correction - dark pixel adjustment ---
    #--- for each variable in the list, reduce by the value of the ref_variable, over target areas ---
    #--- target areas are areas where the frequency of water is ge than the water frequency threshold parameter
    #-------------------------------------------------------------------------------------------------------------
    # --- the 'dp_adjust' dictionary passed in as an argument controls which variables are used as a reference, and which are changed 
    
    for sensor in dp_adjust.keys():
        if sensor in list(instruments.keys()):        
            ref_var = dp_adjust[sensor]['ref_var']
            if not ref_var in ds.data_vars :
                print('variable',ref_var,'expected  but not found in the dataset - correction FAILING')
                return(ds)
            else:
                for target_var in dp_adjust[sensor]['var_list']:
                    if test: print(target_var)
                    if not target_var in ds.data_vars :
                        print('variable',target_var,'expected  but not found in the dataset; terminating the R_correction')
                        return(ds)
                    else:
                        if test : print('calculating values for :',str(target_var+'r'))
                        new_var = str(target_var+'r')
                        ds[new_var] = (ds[target_var] - ds[ref_var]).\
                            where(ds[target_var] > ds[ref_var],0).\
                            where(ds[target_var]>0).\
                            where(ds.wofs_ann_freq>water_frequency_threshold,ds[target_var])
                            # --- previously I used the gap-filled wofs, but this is not conducive to pixel-wise processing ----
                            # where(ds.wofs_fill_freq>water_frequency_threshold,ds[target_var])

    if verbose : #plot graphs of TM illustrating the shift in the distribution
        y = np.random.rand(np.size(ds.msi02_agm))  #creates a random field with the right number of values..
        target_n = 10000
        if np.size(y) > target_n: rand_cut = target_n/np.size(y)
        else: rand_cut = 1
        effective_n = np.size(y[np.argwhere(y<rand_cut)])
        
        if True:
            x = np.arange(0,effective_n)
            plt.plot(np.sort(ds['msi02_agmr'].values.reshape(np.size(ds['msi02_agmr'])))[np.argwhere(y<rand_cut)],x,"b-")
            plt.plot(np.sort(ds['msi02_agm' ].values.reshape(np.size(ds['msi02_agm' ])))[np.argwhere(y<rand_cut)],x,"b--" )
            plt.plot(np.sort(ds['msi03_agmr'].values.reshape(np.size(ds['msi02_agm'])))[np.argwhere(y<rand_cut)],x,"g-")
            plt.plot(np.sort(ds['msi03_agm'].values.reshape(np.size(ds['msi02_agm'])))[np.argwhere(y<rand_cut)],x,"g--")
            plt.plot(np.sort(ds['msi04_agmr'].values.reshape(np.size(ds['msi02_agm'])))[np.argwhere(y<rand_cut)],x,"r-")
            plt.plot(np.sort(ds['msi04_agm'].values.reshape(np.size(ds['msi02_agm'])))[np.argwhere(y<rand_cut)],x,"r--",label='what the ...')
            
            #plt.plot(ds['etm05'].values.reshape(np.size(ds['etm01'])),ds['etm01'].values.reshape(np.size(ds['etm01'])),"o"    
    if verbose : print('R_correction completed normally')
    return(ds)

def water_analysis(ds,
                   water_frequency_threshold= 0.5,
                   wofs_varname             = 'wofs_ann_freq',
                   permanent_water_threshold= 0.875,
                   sigma_coefficient        = 1.2,
                   verbose                  = True,
                   test                     = True):

    # --- extracts permanent water using a threshold on the frequency
    if not wofs_varname in ('wofs_ann_freq','wofs_fill_freq') : 
        print ('INVALID VARIABLE NAME, defaulting to wofs_ann_freq')
        wofs_varname = 'wofs_ann_freq'

    # --- standard deviation of the annual frequency at each pixel - should really be dividing by n-1 but then I would need to change SC ---
    ds['wofs_ann_freq_sigma'] = ((ds.wofs_ann_freq * (1 - ds.wofs_ann_freq)) / ds.wofs_ann_clearcount)**0.5
    ds['wofs_ann_confidence'] = ((1.0 - (ds.wofs_ann_freq_sigma/ds.wofs_ann_freq)) * 100).astype('int16')   
    ds['wofs_pw_threshold']   = (-1 * ds.wofs_ann_freq_sigma * sigma_coefficient) + permanent_water_threshold  #--- threshold varies with p and n 
    ds['wofs_ann_pwater']     = xr.where(ds[wofs_varname]> ds.wofs_pw_threshold,ds[wofs_varname],0)
    ds['wofs_ann_water']      = xr.where(ds[wofs_varname]> water_frequency_threshold,ds[wofs_varname],0)

    # --- A variable called watermask is used in places. I set the value of the mask as sigma or nan --- 
    ds['watermask']  = ds['wofs_ann_freq_sigma'].where(ds[wofs_varname] > water_frequency_threshold)
    
    return(ds)

def rename_vars_robust(dataset,var_names,verbose=False):
        # --- helpful function when renaming variables. var_names nees to be pairs of old and new names  --- 
        for var in var_names :
            if var[0] in np.asarray(dataset.data_vars):
                dataset = dataset.rename_vars({var[0]:var[1]})
        if verbose: 
            print(dataset.data_vars)
        return(dataset)


def set_spacetime_domain(myplace=None,year1='2000',year2='2024',max_cells=1000000,verbose=False,test=False):
    # --- This function sets the space-time domain ---
    # - it contains the parameters for a series of test sites
    # - the input argument tells it which parameters to choose
    # - secondary parameters such as the grid resolution are computed and returned

    if year1 == None: year1 = '2000'
    if year2 == None: year2 = '2024'
    '''
    the structure is:
    - a dictionary of lists, called 'places'. Each list contains:
    -- a dictionary object including x, y, time (which are arrays) which can be read by datacube.load, and
    -- a character list item at the end that is the site description, perhaps not needed.
    Unpacking this needs to be done in the right way to make sense!!
    ''' 

    
    places = {
        'Lake_Baringo'     :   {'run':True, "xyt" :{"x": (36.00,  36.17),     "y": (00.45,00.74),        "time": (year1,year2)},"desc":'Lake Baringo'    },
        'Lake_Tikpan'      :   {'run':True, "xyt" :{"x": ( 1.8215,   1.8265), "y": (6.459,6.4626),       "time": (year1,year2)},"desc":'Lake Tikpan'    },
        'Lake_Chad'        :   {'run':True, "xyt" :{"x": (12.97,  15.50),     "y": (12.40,14.50),        "time": (year1,year2)},"desc":'Lake Chad'       },
        'Weija_Reservoir'  :   {'run':True, "xyt" :{"x": (-0.325, -0.41),     "y": ( 5.54, 5.62),        "time": (year1,year2)  },"desc":''                },
        'Senegal_StLouis'  :   {'run':True, "xyt" :{"x": (-15.74,-15.84),     "y": (16.3, 16.3900),      "time": (year1,year2)  },"desc":'Lac de Guiers'   },
        'Lake_Sulunga'     :   {'run':True, "xyt" :{"x": (34.95, 35.4),       "y": (-6.3, -5.8),         "time": (year1,year2)  },"desc":""                },
        'few_pixels'       :   {'run':False, "xyt" :{"x": (33.1600,33.16005),   "y": (-2.1200, -2.1424),  "time": (year1,year2)  },"desc": ""          },
        'small_area':          {'run':False, "xyt" :{"x": (33.1655,33.1864),    "y": (-2.1532,-2.1444),   "time": (year1,year2)  },"desc": ""          },
        'cameroon_res1'    :   {'run':True, "xyt" :{"y": (6.20,6.30),          "x": (11.25, 11.35),      "time": (year1,year2)  },"desc": "reservoir in cameroon"},
        'Lake_Victoria':       {'run':True, "xyt" :{"x": (33.100,33.300),      "y": (-2.0800,-2.1500),   "time": (year1,year2)  },"desc": ""          },
        'Lake_Mweru':          {'run':True, "xyt" :{"x": (28.200,29.300),      "y": (-9.500,-8.500),     "time": (year1,year2)  },"desc": "Lake Mweru - Zambia / DRC"          },
        'Lake_Mweru_subset':   {'run':True, "xyt" :{"x": (28.450,28.750),      "y": (-9.180,-9.030),     "time": (year1,year2)  },"desc": ""          },
        'Ghana_AwunaBeach':    {'run':True, "xyt" :{"x": (-1.580,-1.640),      "y": (  5.0, 5.05),       "time": (year1,year2)  },"desc": ""          },
        'Ghana_River'     :    {'run':True, "xyt" :{"x": (-1.626,-1.610),      "y": (  5.065,5.089),     "time": (year1,year2)  },"desc": "Ghana, turbid river"          },
        'Large_area':          {'run':False, "xyt" :{"x": ( 32.5,   35.5),      "y" : ( -4.5,-1.5),       "time": (year1,year2)  },"desc": ""          },
        'Lake_vic_west':       {'run':True, "xyt" :{"x": ( 32.5, 32.78),       "y" : ( -2.65,-2.3),      "time": (year1,year2)  },"desc": ""          },
        'Lake_vic_east':       {'run':True, "xyt" :{"x": ( 32.78, 33.3),       "y" : ( -2.65,-2.3),      "time": (year1,year2)  },"desc": ""          },
        'Lake_vic_test':       {'run':True, "xyt" :{"x": ( 32.78, 33.13),       "y" : ( -1.95,-1.6),      "time": (year1,year2)  },"desc": "Lake Victoria cloud affected"},
        'Lake_vic_turbid':     {'run':False, "xyt" :{"x": ( 34.60, 34.70),       "y" : ( -.25,-.20),      "time": (year1,year2)  },"desc": "Lake Victoria turbid area in NE"},
        'Lake_vic_algae':      {'run':True, "xyt" :{"x": ( 34.62, 34.78),       "y" : ( -.18,-.08),      "time": (year1,year2)  },"desc": "Lake Victoria Water Hyacinth affected area in NE, port Kisumu"},
        'Lake_vic_clear':      {'run':True, "xyt" :{"x": ( 34.00, 34.10),       "y" : ( -.32,-.27),      "time": (year1,year2)  },"desc": "Lake Victoria clear water area"},
        'Lake_Victoria_NE' :   {'run':True, "xyt" :{'x': (33.5,34.8),         'y': (-.6,0.4),            'time': (year1,year2)  },"desc": 'Lake Victoria NE'},
        'Morocco':             {'run':True, "xyt" :{"x": (-7.45, -7.65),       "y" : (  32.4,32.5),      "time": (year1,year2)  },"desc": "Barrage Al Massira"          },
        'Thewaterskool_SA':    {'run':True, "xyt" :{"x": (19.1, 19.3),         "y" : ( -34.1  , -33.98), "time": (year1,year2)  },"desc": ""          },
        'SA_dam':              {'run':False, "xyt" :{"x": ( 19.35,   19.47),    "y" : ( -33.800, -33.650),"time": (year1,year2)  },"desc": ""          },
        'SA_dam_north':        {'run':True, "xyt" :{"x": ( 19.42,   19.44),    "y" : ( -33.73 , -33.699),"time": (year1,year2)  },"desc": ""          },
        'SA_dam_south':        {'run':True, "xyt" :{"x": ( 19.415,  19.431),   "y" : ( -33.781, -33.772),"time": (year1,year2)  },"desc": ""          },
        'Ethiopia1of2':        {'run':True, "xyt" :{"x": ( 38.35,   38.65),    "y" : (   7.37 ,   7.55), "time": (year1,year2)  },"desc": "Ethiopia, Shala Hayk'"          },
        'Ethiopia2of2':        {'run':True, "xyt" :{"x": ( 38.66,   38.83),    "y" : (   7.50 ,   7.71), "time": (year1,year2)  },"desc": "Ethiopia, Abyata Hayk'"          },
        'Ethiopia3of2':        {'run':True, "xyt" :{"x": ( 38.50,   38.67),    "y" : (   7.55 ,   7.69), "time": (year1,year2)  },"desc": "Ethiopia, Langano Hayk' (turbid)"          },
        'Ethiopia_Lake_Tana':  {'run':True, "xyt" :{"x": ( 37.05,   37.22),    "y" : (  11.9  ,  12.0),  "time": (year1,year2)  },"desc": "Ethiopia_Lake_Tana"          },
        'Mare_aux_Vacoas':     {'run':True, "xyt" :{"x": ( 57.485,  57.524),   "y" : ( -20.389, -20.359),"time": (year1,year2)  },"desc": "Mare_aux_Vacoas"          },
        'SA_smalldam':         {'run':True, "xyt" :{"x": ( 19.494,  19.498),   "y" : ( -33.802, -33.800),"time": (year1,year2)  },"desc": "Irrigation Dam, South Africa"          },
        'SA_smalldam1':        {'run':True, "xyt" :{"x": ( 19.505, 19.510),   "y" : ( -33.8065, -33.803),"time": (year1,year2)  },"desc": "Irrigation Dam, South Africa, clear water"     },
        'Ethiopia_both':       {'run':False, "xyt" :{"x": ( 38.35,   38.83),    "y" : (   7.37 ,   7.71), "time": (year1,year2)  },"desc": "Ethiopia, Lake Abiata +"          },
        'Lake Chamo'   :       {'run':True, "xyt" :{"x": ( 37.45,   37.65) ,   "y" : (   5.685 ,  5.979), "time": (year1,year2)  },"desc": "Lake Chamo, Ethiopia"          },
        'Lake Ziway'   :       {'run':True, "xyt" :{"x": ( 38.711,  38.966),   "y" : (   7.838 ,  8.148), "time": (year1,year2)  },"desc": "Lake Ziway, Ethiopia"          },
        'Lake Alwassa' :       {'run':True, "xyt" :{"x": ( 38.380,  38.493),   "y" : (   6.977 ,  7.133), "time": (year1,year2)  },"desc": "Lake Alwassa, Ethiopia"          },
        'Lake Elmenteita' :    {'run':True, "xyt" :{"x": ( 36.211,  36.273),   "y" : (  -0.488 , -0.401), "time": (year1,year2)  },"desc": "Lake Elmenteita, Kenya"          },
        'Madagascar':          {'run':True, "xyt" :{"x": ( 43.58 ,  43.76 ),   "y" : ( -22.03 , -21.87 ),"time": (year1,year2)  },"desc": "Farihy Ihotry, Madagascar"          },
        'Lake_Manyara':        {'run':True, "xyt" :{"x": ( 35.724 ,  35.929 ), "y" : ( -03.814, -03.409), "time": (year1,year2) },"desc": "Lake_Manyara, Tanzania"          },#this is the lake to use as an example of monitoring, see 2015-12-28
        'Farihy_':             {'run':True, "xyt" :{"x": ( 43.58 ,  43.76 ),   "y" : ( -22.03 , -21.87 ),"time": (year1,year2)  },"desc": "Farihy Ihotry, Madagascar"          },
        'Farihy_itasy':        {'run':True, "xyt" :{"x": ( 46.73 ,  46.83 ),   "y" : ( -19.10 , -19.04 ),"time": (year1,year2)},"desc": "Farihy Itasy, Madagascar"          },
        'Kolokonda':           {'run':True, "xyt" :{"x": ( 35.4888, 35.5488),   "y" : ( -5.976, -5.916 ),"time": (year1,year2)},"desc": "Kolokonda, Tanzania"          },
        'Dodoma_small':        {'run':True, "xyt" :{"x": ( 35.475 , 35.51),   "y" : ( -6.03, -5.99 )    ,"time": (year1,year2)},"desc": "Dodoma, Tanzania"          },
        'size_test':           {'run':False, "xyt" :{"x": ( 31.400 , 32.40),   "y" : ( -0.00, -1.00 )    ,"time": (year1,year2)  },"desc": "Lake Victoria"          },
        'lake_vic_all':        {'run':False, "xyt" :{"x": ( 31.500 , 34.86),   "y" : ( -3.00, +0.50 )    ,"time": (year1,year2)  },"desc": "Lake Victoria"          },
        'lake_elmenteita':     {'run':True, "xyt" :{"x": ( 36.200 , 36.27),   "y" : ( -0.485, -0.390 )  ,"time": (year1,year2)  },"desc": "Lake Elmenteita"          },
        'mombasa':             {'run':True, "xyt" :{"x": ( 39.500 , 39.72),   "y" : ( -4.10 , -3.97  )  ,"time": (year1,year2)  },"desc": "Mombasa"          },
        'Mauritania_2':        {'run':False, "xyt" :{"x": ( -15.63 , -15.54),   "y" : ( 16.605 , 16.69  ),"time": (year1,year2)},"desc": "Mauritania Wetland"          },
        'Mauritania_1':        {'run':False, "xyt" :{"x": ( -16.37 , -16.32),   "y" : ( 16.41 , 16.45  ) ,"time": (year1,year2)},"desc": "Mauritania Wetland"          },
        'Lake_Nasser_nth':     {'run':True, "xyt" :{"x": (  32.87 ,  32.95),   "y" : ( 23.69 , 23.72  ) ,"time": (year1,year2)},"desc": "Lake Nasser clear water"       },
        'Lake_Nasser_sth':     {'run':True, "xyt" :{"x": (  31.20 ,  31.30),   "y" : ( 21.795, 21.845  ) ,"time": (year1,year2)},"desc": "Lake Nasser turbid water"      },
        'Tana_Hayk'      :     {'run':True, "xyt" :{"x": (  36.95 ,  37.65),   "y" : ( 11.56 , 12.33   ) ,"time": (year1,year2)},"desc": "T'ana Hayk', northern Ethiopia"},
        'Lake_Malawi'    :     {'run':True, "xyt" :{"x": (  34.25 ,  34.97),   "y" : ( -13.6 , -13.3    ) ,"time": (year1,year2)},"desc": "Lake Malawi - part of"},
        'Lago de Cabora' :     {'run':True, "xyt" :{"x": (  30.90 ,  32.52),   "y" : ( -15.95, -15.45  ) ,"time": (year1,year2)},"desc": "Lago de Cabora Basa - Mozambique"},
        'Mtera Reservoir':     {'run':True, "xyt" :{"x": (  35.60 ,  36.01),   "y" : ( - 7.20, - 6.86  ) ,"time": (year1,year2)},"desc": "Lake Nzuhe, Tanzania"},
        'Barrage Joumine':     {'run':True, "xyt" :{"x": (  09.53 ,  09.62),   "y" : ( 36.952,  37.00  ) ,"time": (year1,year2)},"desc": "Joumine Dam,Tunisia"},
        'Tunisia_Dam'    :     {'run':True, "xyt" :{"x": (  08.53 ,  08.56),   "y" : ( 36.685,  36.75  ) ,"time": (year1,year2)},"desc": "Tunisia"},
        'Lake_Ngami'     :     {'run':True, "xyt" :{"x": (  22.55 ,  22.89),   "y" : ( - 20.6, -20.37  ) ,"time": (year1,year2)},"desc": "Botswana"},
        'Lake_Chilwa'    :  {'run':True, "xyt" :{"x": (  35.5 ,  35.9),   "y" : ( - 15.6, -14.90     ) ,"time": (year1,year2)  },"desc": "Malawi - Lake Chilwa"},
        'Lake_Malombe'   :  {'run':True, "xyt" :{"x": (  35.15 ,  35.35),   "y" : ( - 14.8, -14.50   ) ,"time": (year1,year2)  },"desc": "Malawi - Lake Malombe"},
        'Lake_Piti'      :  {'run':True, "xyt" :{"x": (  32.85 ,  32.90),   "y" : ( - 26.6, -26.50   ) ,"time": (year1,year2)  },"desc": "Mozambique - Lake Piti"},
        'Maputo_reserve' :  {'run':True, "xyt" :{"x": (  32.79 ,  32.83),   "y" : ( - 26.55, -26.50  ) ,"time": (year1,year2)  },"desc": "Mozambique - Maputo reserve"},
        'Indian_Ocean'   :  {'run':True, "xyt" :{"x": (  57.75 ,  57.80),   "y" : ( - 20.5 , -20.45  ) ,"time": (year1,year2)  },"desc": "Mauritius - Oceanic waters"},
        'Mare_Vacoas'    :  {'run':True, "xyt" :{"x": (  57.48 ,  57.52),   "y" : ( - 20.38 , -20.36 ) ,"time": (year1,year2)  },"desc": "Mauritius - Mare aux Vacoas"},
        'Naute'          :  {'run':True, "xyt" :{"x": (  17.93 ,  18.05),   "y" : ( - 26.97 , -26.92 ) ,"time": (year1,year2)  },"desc": "Namibia - Naute reserve"},
        'Lake_Turkana'   :  {'run':True, "xyt" :{"x": (  35.80 ,  36.72),   "y" : (    2.38 ,   4.79 ) ,"time": (year1,year2)  },"desc": "Kenya -- Lake Turkana"},
        'Haartbeesport_dam':{'run':True, "xyt" :{"x": (  27.7972, 27.91117), "y": (-25.7761,-25.7275 ) ,"time": (year1,year2)  },"desc": "Haartbeesport Dam  -- South Africa"},
        'Lake Bogoria'   :  {'run':True, "xyt" :{"x": (  36.058, 36.133),   "y" : (  0.1791 ,0.3534  ) ,"time": (year1,year2)  },"desc": "Lake Bogoria -- Tanzania"},
        'HC_1'           :  {'run':False, "xyt" :{"x": (30.50, 30.70    ),  "y" : ( -8.00 ,-7.8      ) ,"time": (year1,year2)  },"desc": "Lake Tanganyika -- Tanzania, low turbidity"},
        'HC_2'           :  {'run':False, "xyt" :{"x": (32.00, 32.20    ),  "y" : ( -8.00 ,-7.80     ) ,"time": (year1,year2)  },"desc": "Lake Rukwa -- Tanzania, very high turbidity"},
        'HC_3'           :  {'run':False, "xyt" :{"x": (32.61, 33.52    ),  "y" : ( -1.469 , -1.00   ) ,"time": (year1,year2)  },"desc": "lake Victoria lake stations -- TZA 32,37,38"},
        'HC_4'           :  {'run':False, "xyt" :{"x": (32.85, 32.87    ),  "y" : ( -2.61 , -2.57    ) ,"time": (year1,year2)  },"desc": "lake Victoria lake stations -- TZA 15 -2.5908	32.86835"},
        'HC_5'           :  {'run':False, "xyt" :{"x": (-6.7502, -6.7473),  "y" : ( 33.9308 , 33.9347) ,"time": (year1,year2)  },"desc": "Morocco near Rabbat -- MAR 00002  -6.75 33.93333 "},
        'HC_6'           :  {'run':False, "xyt" :{"x": (-7.6360, -7.6308),  "y" : ( 32.4727 , 33.4758) ,"time": (year1,year2)  },"desc": "Morocco -- MAR 00003  -6.75 32.93333 "},
        }

     #Manyara is a shallow alkaline lake 10 feet deep. https://wildlifesafaritanzania.com/facts-about-lake-manyara-national-park/
    #During sampling periods[2006-2008], big blooms of blue-green algae were observed covering thewhole 
    #water surface especially in March, April, May and June. 
    #Most of the blooms were along the lake shore, giving a characteristic foul smell and foam 
    #https://www.researchgate.net/publication/266095025_Assessment_of_farming_practices_and_uses_of_agrochemicals_in_Lake_Manyara_basin_Tanzania
    #(6) (PDF) Assessment of farming practices and uses of agrochemicals in Lake Manyara basin, Tanzania. Available from:     https://www.researchgate.net/publication/266095025_Assessment_of_farming_practices_and_uses_of_agrochemicals_in_Lake_Manyara_basin_Tanzania [accessed Sep 07 2024]

    #Sulunga is a higly variable waterbody. annual wofs will be needed to work with this. 
    #'few_pixels' is useful for debugging and exploration
    
    if myplace == None:
        return places
    
    #Extract the key parameters for further use:
    #First check the name; places.keys will list the key values, boolean operators do the rest!
    if(not myplace in places.keys()):
        print('INVALID AOI! :- ',myplace,'\n Valid values are:')
        for name in places.keys(): print(name)
        return('','','','','','')  #null values here still need to match the spaces of the expected output.

    AOI = places[myplace]['xyt'] #extract the site of interest, AOI is now a list
    if False:
        print((AOI))
        print((AOI['x'])[0])  #prints the first x coordinate, which comes from the dictionary object, which is the first thing in the list called 'here'. 

    spacetime_domain = AOI
    file_name        = myplace
    site_name        = places[myplace]["desc"]   #should return the description string

    #establish a reasonable grid resolution between min and max based on the AOI
    cell_min = 10
    cell_max = 500
    x0 = AOI["x"][0]        
    x1 = AOI["x"][1]
    y0 = AOI["y"][0]
    y1 = AOI["y"][1]
    dxm = ((x1-x0) * 100000) * np.cos(y0 * np.pi/180.0) #metres approx allowing for lattitudinal compression of dx
    dym = ((y1-y0) * 100000)  #metres approx
    dAm = abs(dxm * dym)           #sqare metres total extent
    #max_cells = 100000       # grid cells covering the extent, about 100km2 at 10m resolution
    cell_m2 = dAm / max_cells
    cell_dxm = cell_m2**0.5

    if  cell_dxm < cell_min :
        cell_dxm = cell_min
    else :
        if  cell_dxm > cell_max :
            cell_dxm = cell_max
    cell_dxm         = int(cell_dxm / 10) * 10
    grid_resolution  = (cell_dxm,cell_dxm) 
    cellcount        = dAm / (cell_dxm**2)

    aspect_ratio = np.abs(dym/dxm)
    cell_area = (cell_dxm**2)/1000000 #km^2

    #apply a bilinear interpolation if cells that are small relative to the pixels
    if cell_dxm > 60 :
        resampling_option = "nearest"
    else:
        resampling_option = "bilinear"

    year1 = AOI["time"][0]
    year2 = AOI["time"][1]
        
    if verbose:
        print(myplace,"AOI: ",spacetime_domain, places[myplace]['desc'])
        print("Grid resolution will be:", grid_resolution)
        print("Rough dimensions (x,y): ", int(dxm/1000), " by " ,int(abs(dym/1000)), "kilometres")
        print("Total cells is roughly: ", int(cellcount))
        print("Cell area is: ", cell_area," km2")
        print("Resampling :",resampling_option)
        print("Site name: ", site_name)
        print("Years:" , year1, year2)

    return(spacetime_domain,grid_resolution,cell_area,resampling_option,year1,year2)




