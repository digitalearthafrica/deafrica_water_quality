
import xarray as xr
import pandas as pd
from importlib.resources import files
import scipy  as sp

def harmonise_for_instrument(input_wq_ds: xr.Dataset,          
                             varname: str ,     
                             test = False,   # test flag; this also changes the behaviour of the function 
                            ): 
    # TODO check if all vars can be done at once for efficiency
    input_variables = list(input_wq_ds["tsm_chla_tsi"].data_vars)
    # running single variable harmonization; for testing purposes only

    # TODO: If very large, save to s3 bucket (public) and read from there. 
    tookup_table_fp = files("water_quality.data").joinpath('targ_ref_lookup.csv')
    lookup_table = pd.read_csv(tookup_table_fp,index_col=0)

    # the distributions data has the statistical distributions of all variables, 
    # calculated on all images (not geomedians) for 2019 and 2020 over about 20 sites.

    # TODO: If very large, save to s3 bucket (public) and read from there. 
    distributions_fp = files("water_quality.data").joinpath("measurment_distributions.nc")
    distributions =  xr.open_dataset(distributions_fp).load(); distributions.close()

    # --- find the target variable
    for varname in input_variables:
        if not varname in list(lookup_table.target):
            print(varname+' not found, data unchanged')
            return(input_wq_ds)
 
    refname = lookup_table[lookup_table.target==varname]['reference'].item()   
    if test: print(varname,refname)
    if refname == varname:
        if test: print('data will be unchanged')
        return(input_wq_ds)

    Q              = distributions.q 
    method = 'quadratic'    # quadratic may be okay but should not be necessary and can introduce problems at the distribution tails
    method = 'linear'    

    # i,j,k,l control which parts of the distribution we  mnodel - conclude after experimentation that issues arise if the full
    #         distribution is not used
    
    # --- fit a model to X; ie Q = f(x); limit the valid range of the function with excess values being set to 0 or 1
    i,j = 0,101  
    f = sp.interpolate.interp1d(distributions[varname][i:j],Q[i:j],kind=method,bounds_error = False,fill_value = (0,1))    

    # --- fit an inverse model to Y; ie Y = g(Q). 
    k,l = 0,101  
    g = sp.interpolate.interp1d(Q[k:l],distributions[refname][k:l],kind=method,bounds_error=False)

    # --- given the functions f and g, the required adjustment is g(f(x)) ----------------
    
    # --- create a new variable as a data array that matches the one provided, so that we can return a nice array
    if test and False:
        print('returning functions as well as data')
        return(
            xr.DataArray(data=g(f(input_wq_ds)), dims=input_wq_ds.dims,coords = input_wq_ds.coords,name = input_wq_ds.name+'_h'),f,g
        )    # --- create a new variable as a data array that matches the one provided, so that we can return a nice array
    return(
        xr.DataArray(data=g(f(input_wq_ds)), dims=input_wq_ds.dims,coords = input_wq_ds.coords,name = input_wq_ds.name+'_h')
        )
