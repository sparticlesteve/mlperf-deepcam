import os
import numpy as np
import h5py as h5

#merge function helper
def merge_token(token1, token2):
    #extract data
    #first
    n1 = token1[0]
    dmean1 = token1[1]
    dsqmean1 = token1[2]
    dmin1 = token1[3]
    dmax1 = token1[4]
    #second
    n2 = token2[0]
    dmean2 = token2[1]
    dsqmean2 = token2[2]
    dmin2 = token2[3]
    dmax2 = token2[4]

    #create new token
    nres = n1 + n2
    dmeanres = float(n1)/float(nres)*dmean1 + float(n2)/float(nres)*dmean2
    dsqmeanres = float(n1)/float(nres)*dsqmean1 + float(n2)/float(nres)*dsqmean2
    dminres = np.minimum(dmin1, dmin2)
    dmaxres = np.maximum(dmax1, dmax2)

    return (nres, dmeanres, dsqmeanres, dminres, dmaxres)


#create data token
def create_token(filename, data_format="nchw"):
    
    with h5.File(filename, "r") as f:
        arr = f["climate/data"][...]
    
    #prep axis for ops
    axis = (1,2) if data_format == "nchw" else (0,1)

    #how many samples do we have: just 1 here
    n = 1
    #compute stats
    mean = np.mean(arr, axis=axis)
    meansq = np.mean(np.square(arr), axis=axis)
    minimum = np.amin(arr, axis=axis)
    maximum = np.amax(arr, axis=axis)

    #result
    result = (n, mean, meansq, minimum, maximum)
    
    return result
        

#global parameters
overwrite = False
data_format = "nhwc"
data_path_prefix = "/raid/tkurth/cam5_data"

#root path
root = os.path.join( data_path_prefix, "train" )
            
files = [ os.path.join(root, x)  for x in os.listdir(root) \
                  if x.endswith('.h5') and x.startswith('data-') ]

#get first token and then merge recursively
token = create_token(files[0], data_format)
for filename in files[1:]:
    token = merge_token(create_token(filename, data_format), token)

#save the stuff
with h5.File(os.path.join(data_path_prefix, "stats.h5"), "w") as f:
    f["climate/count"]=token[0]
    f["climate/mean"]=token[1]
    f["climate/sqmean"]=token[2]
    f["climate/minval"]=token[3]
    f["climate/maxval"]=token[4]