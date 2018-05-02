#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, sys, numpy as np, subprocess as sp
from skimage.external import tifffile
from scipy.stats import wasserstein_distance
import numpy as np



def register_volumes_to_seed(input_folder, output_folder, parameters, seed, verbose = True):
    '''Main function to run registration on a tifffiles

    Inputs
    -------------------
    input_folder = folder containing tiff volumes
    output_folder = place to save registration data
    parameters = list of paths to parameter files for registration in the *order they should be applied*  
    seed = path to tifffile that will be used a fixed image for registration

    '''
    
    #make output folder:
    makedir(output_folder)
    
    #find all brains
    brains = listdirfull(input_folder)
    
    #remove seed
    brains = [xx for xx in brains if xx != seed]
    
    if verbose: sys.stdout.write('{} Volume(s) found, registering each to seed {}.\nInput folder {}\nOutput folder {}\nParameter files {}\n'.format(len(brains), seed, input_folder, output_folder, parameters)); sys.stdout.flush()
    
    #run
    for brain in brains:
        
        #run registration on each brain, this can take time. Go grab a coffee (or two)
        out = os.path.join(output_folder, os.path.basename(brain)[:-4]); makedir(out)
        if verbose: sys.stdout.write('\nStarting registation on {}...'.format(os.path.basename(brain))); sys.stdout.flush()
        
        #run
        elastix_command_line_call(fx=seed, mv=brain, out=out, parameters=parameters, fx_mask=False)
        if verbose: sys.stdout.write('completed.'.format(os.path.basename(brain))); sys.stdout.flush()
        
    if verbose: sys.stdout.write('\nCompleted Regstration :] '.format(os.path.basename(brain))); sys.stdout.flush()
    return

def generate_median_image(output_folder, parameters, memmappth, dst, verbose = True):
    '''Function to collect post-registered volumes, generate a memory mapped array and then save out median volume
    '''    

    if verbose: sys.stdout.write('Collecting data and generating memory mapped array'); sys.stdout.flush()
    nm = 'result.{}.tif'.format(len(parameters)-1)
    brains = [os.path.join(xx, nm) for xx in listdirfull(output_folder) if os.path.exists(os.path.join(xx, nm))]
    vol = tifffile.imread(brains[0])
    z,y,x = vol.shape
    dtype = vol.dtype
    
    #init array
    arr = load_memmap_arr(memmappth, mode='w+', shape = (len(brains),z,y,x), dtype = dtype)
    
    #load
    for i, brain in enumerate(brains):
        arr[i] = tifffile.imread(brain)
        arr.flush()
    if dst[-4:] != '.tif': dst = dst+'.tif'
    if verbose: sys.stdout.write('...completed\nTaking median and saving as {}'.format(dst)); sys.stdout.flush()
    
    #median volume
    vol = np.median(arr, axis=0)
    tifffile.imsave(dst, vol)
    if verbose: sys.stdout.write('...completed'); sys.stdout.flush()
    return



def elastix_command_line_call(fx, mv, out, parameters, fx_mask=False, verbose=False):
    '''Wrapper Function to call elastix using the commandline, this can be time consuming
    
    Inputs
    -------------------
    fx = fixed path (usually Atlas for 'normal' noninverse transforms)
    mv = moving path (usually volume to register for 'normal' noninverse transforms)
    out = folder to save file
    parameters = list of paths to parameter files IN ORDER THEY SHOULD BE APPLIED
    fx_mask= (optional) mask path if desired
    
    Outputs
    --------------
    ElastixResultFile = '.tif' or '.mhd' result file
    TransformParameterFile = file storing transform parameters
    
    '''
    e_params=['elastix', '-f', fx, '-m', mv, '-out', out]
    if fx_mask: e_params=['elastix', '-f', fx, '-m', mv, '-fMask', fx_mask, '-out', out]
    
    ###adding elastix parameter files to command line call
    for x in range(len(parameters)):
        e_params.append('-p')
        e_params.append(parameters[x])
    
    #set paths
    TransformParameterFile = os.path.join(out, 'TransformParameters.{}.txt'.format((len(parameters)-1)))
    ElastixResultFile = os.path.join(out, 'result.{}.tif'.format((len(parameters)-1)))
    
    #run elastix: 
    try:                
        if verbose: print ('Running Elastix, this can take some time....\n')
        sp.call(e_params)
        if verbose: print('Past Elastix Commandline Call')
    except RuntimeError, e:
        print('\n***RUNTIME ERROR***: {} Elastix has failed. Most likely the two images are too dissimiliar.\n'.format(e.message))
        pass      
    if os.path.exists(ElastixResultFile) == True:    
        if verbose: print('Elastix Registration Successfully Completed\n')
    #check to see if it was MHD instead
    elif os.path.exists(os.path.join(out, 'result.{}.mhd'.format((len(parameters)-1)))) == True:    
        ElastixResultFile = os.path.join(out, 'result.{}.mhd'.format((len(parameters)-1)))
        if verbose: print('Elastix Registration Successfully Completed\n')
    else:
        print ('\n***ERROR***Cannot find elastix result file, try changing parameter files\n: {}'.format(ElastixResultFile))
        return

        
    return ElastixResultFile, TransformParameterFile



def makedir(path):
    '''Simple function to make directory if path does not exists'''
    if os.path.exists(path) == False:
        os.mkdir(path)
    return

def listdirfull(x, keyword=False):
    '''might need to modify based on server...i.e. if automatically saving a file called 'thumbs'
    '''
    if not keyword:
        lst = [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != '.' and '~' not in xx and 'Thumbs.db' not in xx]
    else:
        lst = [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != '.' and '~' not in xx and 'Thumbs.db' not in xx and keyword in xx]

    lst.sort()
    return lst


def load_memmap_arr(pth, mode='r', dtype = 'uint16', shape = False):
    '''Function to load memmaped array.

    Inputs
    -----------
    pth: path to array
    mode: (defaults to r)
    +------+-------------------------------------------------------------+
    | 'r'  | Open existing file for reading only.                        |
    +------+-------------------------------------------------------------+
    | 'r+' | Open existing file for reading and writing.                 |
    +------+-------------------------------------------------------------+
    | 'w+' | Create or overwrite existing file for reading and writing.  |
    +------+-------------------------------------------------------------+
    | 'c'  | Copy-on-write: assignments affect data in memory, but       |
    |      | changes are not saved to disk.  The file on disk is         |
    |      | read-only.                                                  |
    dtype: digit type
    shape: (tuple) shape when initializing the memory map array

    Returns
    -----------
    arr
    '''
    if shape:
        assert mode =='w+', 'Do not pass a shape input into this function unless initializing a new array'
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr

def mse(arr0, arr1):
    '''
    '''
    return np.square(np.subtract(arr0, arr1)).mean()
    


def make_histogram(arr0, arr1, bins = 200):
  '''
  '''
  minval = int(min(np.min(arr0), np.min(arr1)))
  maxval = int(max(np.max(arr0), np.max(arr1)))
  bins = range(minval, maxval, (maxval-minval)/bins)
  h0 = np.histogram(arr0, bins=bins)[0] / float(arr0.ravel().shape[0])
  h1 = np.histogram(arr1, bins=bins)[0] / float(arr1.ravel().shape[0])
  
  return h0, h1

if __name__ == '__main__':
	arr0 = np.random.randint(0,100,(200,200,200))
	arr1 = np.random.randint(50,255,(200,200,200))
	h0, h1 = make_histogram(arr0, arr1, bins = 200)
	dist = wasserstein_distance(h0, h1)
	print(dist)

	import seaborn as sns
	sns.jointplot(h0, h1, kind='kde')


	for i, src in enumerate(arr):
	    src = rotate(np.swapaxes(src, 0,1), 270, axes=(1,2))[::-1]
	    src = np.copy(src)*1.0
	    src = skimage.exposure.adjust_gamma(src, gamma=.7, gain=3)
	    
	    #convert to RGB
	    src = np.stack([src, src, src],axis=-1).astype(np.uint8)
	    
	    skvideo.io.vwrite(os.path.join(dst, str(i).zfill(4)+'.mp4'), src) #
	    print i

