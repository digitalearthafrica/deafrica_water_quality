#!/usr/bin/env python
# coding: utf-8
import numpy  as np
import xarray as xr
import matplotlib.pyplot as plt
import gc   # garbage collection


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
def ChlA_MERIS2B(dataset, band_708, band_665, verbose=False):   #matching MSI bands are 5 and 4
# ---- Functions to estimate ChlA using the MERIS and MODIS 2-Band models, using closest bands from MSI (6, 5, 4) or other
    if verbose: print("ChlA_MERIS two-band model")
    X = dataset[band_708] / dataset[band_665]
    return  (25.28 * (X)*2) + 14.85 * (X) - 15.18


def ChlA_MODIS2B(dataset, band_748, band_667,verbose=False):  #matching MSI bands are 6 and 4
    if verbose: print("ChlA_MODIS two-band model")
    X = dataset[band_748] / dataset[band_667]
    return (190.34 * X) - 32.45

# ---- Normalised Difference Suspeneded Sediment Index NDSSI ---
#    These are essentially
#        red-green / red_+green, or,
#        blue-NIR / blue + NIR

def NDSSI_RG(dataset, red_band, green_band,verbose=False):
    if verbose: print("NDSSI_RG")
    return ((dataset[red_band] - dataset[green_band]) / (dataset[red_band] + dataset[green_band]))

def NDSSI_BNIR(dataset, blue_band, NIR_band,verbose=False):
    if verbose: print("NDSSI_BNIR")
    return ((dataset[blue_band] - dataset[NIR_band]) / (dataset[blue_band] + dataset[NIR_band]))

# ---- Turbidity index of Yu, X. et al.
#    An empirical algorithm to seamlessly retrieve the concentration of suspended particulate matter 
#    from water color across ocean to turbid river mouths. Remote Sens. Environ. 235, 111491 (2019).
#    Used in screening turbid waters for mapping floating algal blooms
#    Initially developed with TM
#    -TI = ((Red − green) − (NIR − Rgreen)) ^ 0.5

def TI_yu(dataset,NIR,Red,Green,scalefactor=0.01,verbose=False):
    if verbose: print('TI_yu')
    return  scalefactor * ( ( (dataset[Red] - dataset[Green]) - (dataset[NIR] - dataset[Green]) ) ** 0.5 )
#    return  scalefactor * ((dataset[Red] - dataset[Green]) - ((dataset[NIR] - dataset[Green]) * 0.5)) correction!

# ---- Lymburner Total Suspended Matter (TSM)
# Paper: [Lymburner et al. 2016](https://www.sciencedirect.com/science/article/abs/pii/S0034425716301560)
# Units of mg/L concentration. Variants for ETM and OLT, slight difference in parameters.
# These models, developed by leo lymburner and arnold dekker, are simple, stable, and produce credible # 
# results over a range of observations

def TSM_LYM_ETM(dataset, green_band, red_band, scale_factor=0.0001,verbose=False):
    if verbose: print("TSM_LYM_ETM")
    return 3983 * ((dataset[green_band] + dataset[red_band]) * scale_factor / 2) ** 1.6246

def TSM_LYM_OLI(dataset, green_band, red_band, scale_factor=0.0001,verbose=False):
        if verbose: print("TSM_LYM_OLI")
        return 3957 * ((dataset[green_band] + dataset[red_band]) * scale_factor / 2) ** 1.6436

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

# ---- Quang Total Suspended Solids (TSS)
# Paper: Quang et al. 2017
# Units of mg/L concentration
def TSS_QUANG8(dataset,red_band,verbose=False):
    # ---- Function to calculate quang8 value ----
    if verbose: print("TSS_QUANG8")
    return 380.32 * (dataset.red_band) * 0.0001 - 1.7826

# ---- Model of Zhang et al 2023: ----
#      This model seems to be funamentally unstable, using band ratios as an exponent 
#      is asking for extreme values and numerical overflows. It runs to inf all over the place.
#      Green = B3 520-600; Blue = B2 450-515 Red = B4 630-680
#  The function is not scale-less.
#  The factor of 0.0001 is not part of the forumula but  scales back from 10000 range which is clearly 
#  ridiculous (exp(10000) etc. is not a good number). This therefore avoids overflow. 
# This model can only be used together with other models and indices; it may handle some situations well...

def TSS_Zhang(dataset, blue_band, green_band, red_band, scale_factor=0.0001,verbose=False):
    if verbose: print("TSS_Zhang")
    abovezero = .00001  #avoids div by zero if blue is zero
    GplusR = dataset[green_band] + dataset[red_band]             
    RdivB  = dataset[red_band] / (dataset[blue_band] + abovezero)   
    X = (GplusR * RdivB) * scale_factor                          
    #return(10**(14.44*X))*1.20
    #return  np.exp(14.44* X)*1.20
    return  (14.44* X)      #the distribution of results is exponential; this measure will be more stable without raising to the power.


def OWT_pixel(ds,instrument,water_frequency_threshold=0.8,resample_rate=3,verbose=True, test=True):
    # --- Determine the Open Water Type for each pixel, over areas that are usually water. 
    # --- 'instrument' is a dictionary established while building the dataset
    # --- 'resample_rate' is the spatial resample step to reduce the memory required
    # --- The returned lattice is on the same t,x,y coordinates as the original, after resampling back to it
    # --- this code will need revisiting to support non-geomedian data which has more bands available. 
    # --- Also, right now, msi and oli are dealt with separately rather than as alternatives...
    # --- it would be nice to remove this 'hard-wiring'.
    # --- Memory intensive due to the use of vector multiplication (dot products). 
    #      A less elegant coding approch might be more memory efficient ----
    # 
    
    if verbose: print('Determining the Optical Water Type ...')
    owt_groups = {}
    owt_groups['oligotrophic']                   = [3,9,13]
    owt_groups['eutropic and blue-green']        = [1,2,4,5,11,12]
    owt_groups['hypereutrophic and green-brown'] = [6,7,8,10]

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

def hue_adjust(dataset) :
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

def hue_calculation(dataset,instrument='msi_agm',test=False,verbose=False) : 
    #---- the hue is calculated by conversion of the wavelengths to chromatic coordinates using sensor-specific coefficients
    
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

def R_correction(ds,dp_adjust,instruments,water_frequency_threshold=0.9,verbose=False,test=True):
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

    '''
    the structure is:
    - a dictionary of lists, called 'places'. Each list contains:
    -- a dictionary object including x, y, time (which are arrays) which can be read by datacube.load, and
    -- a character list item at the end that is the site description, perhaps not needed.
    Unpacking this needs to be done in the right way to make sense!!
    ''' 

    
    places = {
        'Lake_Baringo'     :   {'run':True, "xyt" :{"x": (36.00,  36.17),     "y": (00.45,00.74),        "time": (year1,year2)},"desc":'Lake Baringo'    },
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
        'Lake_vic_turbid':     {'run':True, "xyt" :{"x": ( 34.60, 34.70),       "y" : ( -.25,-.20),      "time": (year1,year2)  },"desc": "Lake Victoria turbid area in NE"},
        'Lake_vic_clear':      {'run':True, "xyt" :{"x": ( 34.00, 34.10),       "y" : ( -.32,-.27),      "time": (year1,year2)  },"desc": "Lake Victoria clear water area"},
        'Lake_Victoria_NE' :   {'run':True, "xyt" :{'x': (33.5,34.8),         'y': (-.6,0.4),            'time': (year1,year2)  },"desc": 'Lake Victoria NE'},
        'Morocco':             {'run':True, "xyt" :{"x": (-7.45, -7.65),       "y" : (  32.4,32.5),      "time": (year1,year2)  },"desc": "Barrage Al Massira"          },
        'Thewaterskool_SA':    {'run':True, "xyt" :{"x": (19.1, 19.3),         "y" : ( -34.1  , -33.98), "time": (year1,year2)  },"desc": ""          },
        'SA_dam':              {'run':True, "xyt" :{"x": ( 19.35,   19.47),    "y" : ( -33.800, -33.650),"time": (year1,year2)  },"desc": ""          },
        'SA_dam_north':        {'run':True, "xyt" :{"x": ( 19.42,   19.44),    "y" : ( -33.73 , -33.699),"time": (year1,year2)  },"desc": ""          },
        'SA_dam_south':        {'run':True, "xyt" :{"x": ( 19.415,  19.431),   "y" : ( -33.781, -33.772),"time": (year1,year2)  },"desc": ""          },
        'Ethiopia1of2':        {'run':True, "xyt" :{"x": ( 38.35,   38.65),    "y" : (   7.37 ,   7.55), "time": (year1,year2)  },"desc": "Ethiopia, Shala Hayk'"          },
        'Ethiopia2of2':        {'run':True, "xyt" :{"x": ( 38.66,   38.83),    "y" : (   7.50 ,   7.71), "time": (year1,year2)  },"desc": "Ethiopia, Abyata Hayk'"          },
        'Ethiopia3of2':        {'run':True, "xyt" :{"x": ( 38.50,   38.67),    "y" : (   7.55 ,   7.69), "time": (year1,year2)  },"desc": "Ethiopia, Langano Hayk' (turbid)"          },
        'Ethiopia_Lake_Tana':  {'run':True, "xyt" :{"x": ( 37.05,   37.22),    "y" : (  11.9  ,  12.0),  "time": (year1,year2)  },"desc": "Ethiopia_Lake_Tana"          },
        'Mare_aux_Vacoas':     {'run':True, "xyt" :{"x": ( 57.485,  57.524),   "y" : ( -20.389, -20.359),"time": (year1,year2)  },"desc": "Mare_aux_Vacoas"          },
        'SA_smalldam':         {'run':True, "xyt" :{"x": ( 19.494,  19.498),   "y" : ( -33.802, -33.800),"time": (year1,year2)  },"desc": "Irrigation Dam, South Africa"          },
        'Ethiopia_both':       {'run':False, "xyt" :{"x": ( 38.35,   38.83),    "y" : (   7.37 ,   7.71), "time": (year1,year2)  },"desc": "Ethiopia, Lake Abiata +"          },
        'Madagascar':          {'run':True, "xyt" :{"x": ( 43.58 ,  43.76 ),   "y" : ( -22.03 , -21.87 ),"time": (year1,year2)  },"desc": "Farihy Ihotry, Madagascar"          },
        'Lake_Manyara':        {'run':True, "xyt" :{"x": ( 35.68 ,  35.92 ),   "y" : ( -03.86 , -03.38), "time": (year1,year2) },"desc": "Lake_Manyara, Tanzania"          },#this is the lake to use as an example of monitoring, see 2015-12-28
        'Farihy_':             {'run':True, "xyt" :{"x": ( 43.58 ,  43.76 ),   "y" : ( -22.03 , -21.87 ),"time": (year1,year2)  },"desc": "Farihy Ihotry, Madagascar"          },
        'Farihy_itasy':        {'run':True, "xyt" :{"x": ( 46.73 ,  46.83 ),   "y" : ( -19.10 , -19.04 ),"time": (year1,year2)},"desc": "Farihy Itasy, Madagascar"          },
        'Kolokonda':           {'run':True, "xyt" :{"x": ( 35.4888, 35.5488),   "y" : ( -5.976, -5.916 ),"time": (year1,year2)},"desc": "Kolokonda, Tanzania"          },
        'Dodoma_small':        {'run':True, "xyt" :{"x": ( 35.475 , 35.51),   "y" : ( -6.03, -5.99 )    ,"time": (year1,year2)},"desc": "Dodoma, Tanzania"          },
        'size_test':           {'run':False, "xyt" :{"x": ( 31.400 , 32.40),   "y" : ( -0.00, -1.00 )    ,"time": (year1,year2)  },"desc": "Lake Victoria"          },
        'lake_vic_all':        {'run':False, "xyt" :{"x": ( 31.500 , 34.86),   "y" : ( -3.00, +0.50 )    ,"time": (year1,year2)  },"desc": "Lake Victoria"          },
        'lake_elmenteita':     {'run':True, "xyt" :{"x": ( 36.200 , 36.27),   "y" : ( -0.485, -0.390 )  ,"time": (year1,year2)  },"desc": "Lake Elmenteita"          },
        'mombasa':             {'run':True, "xyt" :{"x": ( 39.500 , 39.72),   "y" : ( -4.10 , -3.97  )  ,"time": (year1,year2)  },"desc": "Mombasa"          },
        'Mauritania_2':        {'run':True, "xyt" :{"x": ( -15.63 , -15.54),   "y" : ( 16.605 , 16.69  ),"time": (year1,year2)},"desc": "Mauritania Wetland"          },
        'Mauritania_1':        {'run':True, "xyt" :{"x": ( -16.37 , -16.32),   "y" : ( 16.41 , 16.45  ) ,"time": (year1,year2)},"desc": "Mauritania Wetland"          },
        'Lake_Nasser_nth':     {'run':True, "xyt" :{"x": (  32.87 ,  32.95),   "y" : ( 23.69 , 23.72  ) ,"time": (year1,year2)},"desc": "Lake Nasser clear water"       },
        'Lake_Nasser_sth':     {'run':True, "xyt" :{"x": (  31.20 ,  31.30),   "y" : ( 21.795, 21.845  ) ,"time": (year1,year2)},"desc": "Lake Nasser turbid water"      },
        'Tana_Hayk'      :     {'run':True, "xyt" :{"x": (  36.95 ,  37.65),   "y" : ( 11.56 , 12.33   ) ,"time": (year1,year2)},"desc": "T'ana Hayk', northern Ethiopia"},
        'Lake_Malawi'    :     {'run':True, "xyt" :{"x": (  34.25 ,  34.97),   "y" : ( -13.6 , -13.3    ) ,"time": (year1,year2)},"desc": "Lake Malawi - part of"},
        'Lago de Cabora' :     {'run':True, "xyt" :{"x": (  30.90 ,  32.52),   "y" : ( -15.95, -15.45  ) ,"time": (year1,year2)},"desc": "Lago de Cabora Basa - Mozambique"},
        'Mtera Reservoir':     {'run':True, "xyt" :{"x": (  35.60 ,  36.01),   "y" : ( - 7.20, - 6.86  ) ,"time": (year1,year2)},"desc": "Lake Nzuhe, Tanzania"},
        'Barrage Joumine':     {'run':True, "xyt" :{"x": (  09.53 ,  09.62),   "y" : ( 36.952,  37.00  ) ,"time": (year1,year2)},"desc": "Joumine Dam,Tunisia"},
        'Tunisia_Dam'    :     {'run':True, "xyt" :{"x": (  08.53 ,  08.56),   "y" : ( 36.685,  36.75  ) ,"time": (year1,year2)},"desc": "Tunisia"},
        'Lake_Ngami'     :     {'run':True, "xyt" :{"x": (  22.55 ,  22.89),   "y" : ( - 20.6, -20.37  ) ,"time": (year1,year2)},"desc": "Botswana"},
        'Lake_Chilwa'    :     {'run':True, "xyt" :{"x": (  35.5 ,  35.9),   "y" : ( - 15.6, -14.90  ) ,"time": (year1,year2)},"desc": "Malawi - Lake Chilwa"},
        'Lake_Malombe'   :     {'run':True, "xyt" :{"x": (  35.15 ,  35.35),   "y" : ( - 14.8, -14.50  ) ,"time": (year1,year2)},"desc": "Malawi - Lake Malombe"},
        'Lake_Piti'      :     {'run':True, "xyt" :{"x": (  32.85 ,  32.90),   "y" : ( - 26.6, -26.50  ) ,"time": (year1,year2)},"desc": "Mozambique - Lake Piti"},
        'Maputo_reserve' :     {'run':True, "xyt" :{"x": (  32.79 ,  32.83),   "y" : ( - 26.55, -26.50  ) ,"time": (year1,year2)},"desc": "Mozambique - Maputo reserve"},
        'Indian_Ocean'   :     {'run':True, "xyt" :{"x": (  57.75 ,  57.80),   "y" : ( - 20.5 , -20.45  ) ,"time": (year1,year2)},"desc": "Mauritius - Oceanic waters"},
        'Mare_Vacoas'    :     {'run':True, "xyt" :{"x": (  57.48 ,  57.52),   "y" : ( - 20.38 , -20.36  ) ,"time": (year1,year2)},"desc": "Mauritius - Mare aux Vacoas"},
        'Naute'          :     {'run':True, "xyt" :{"x": (  17.93 ,  18.05),   "y" : ( - 26.97 , -26.92  ) ,"time": (year1,year2)},"desc": "Namibia - Naute reserve"},
        'Lake_Turkana'   :     {'run':True, "xyt" :{"x": (  35.80 ,  36.72),   "y" : (    2.38 ,   4.79  ) ,"time": (year1,year2)},"desc": "Kenya -- Lake Turkana"},
        }

     #Manyara is a shallow alkaline lake 10 feet deep. https://wildlifesafaritanzania.com/facts-about-lake-manyara-national-park/
    #During sampling periods[2006-2008], big blooms of blue-green algae were observed covering thewhole 
    #water surface especially in March, April, May and June. 
    #Most of the blooms were along the lake shore, giving a characteristic foul smell and foam 
    #https://www.researchgate.net/publication/266095025_Assessment_of_farming_practices_and_uses_of_agrochemicals_in_Lake_Manyara_basin_Tanzania
    #(6) (PDF) Assessment of farming practices and uses of agrochemicals in Lake Manyara basin, Tanzania. Available from: https://www.researchgate.net/publication/266095025_Assessment_of_farming_practices_and_uses_of_agrochemicals_in_Lake_Manyara_basin_Tanzania [accessed Sep 07 2024]

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
    x0 = AOI["x"][0]        #the brackets here are not needed, but help me to keep track!
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




