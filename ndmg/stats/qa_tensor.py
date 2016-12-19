from dipy.reconst.dti import fractional_anisotropy, color_fa
from argparse import ArgumentParser
from scipy import ndimage

import os
import re
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # very important above pyplot import
import matplotlib.pyplot as plt
import sys

def tensor2fa(fs, outdir=None):
    '''
    fs: list of 2-tuples containing the tensor file and dti file
    outdir: location of output directory. default is none (folder named fa_maps is created)
    '''
    print(fs)
    for files in fs:
        print(files)
        ften, fdti = files
        img = nib.load(fdti)    
        dti_data = img.get_data()
        
        with np.load(ften) as data:
            tensor_fit = data['arr_0']
            tensor_fit = tensor_fit.tolist()
            
        FA = fractional_anisotropy(tensor_fit.evals)
        # get rid of NaNs
        FA[np.isnan(FA)] = 0
        # check if outfile and outdir are defined
        outfile = os.path.split(ften)[1]
        outfile = outfile.split(".")[0]
        if outdir == None:
            outdir = './'
        # generate the RGB FA map
        FA = np.clip(FA, 0, 1)
        RGB = color_fa(FA, tensor_fit.evecs)
        # save the RGB FA map
        nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), img.affine), outdir + outfile + '_fa_rgb.nii.gz')



def save_fa_pngs(path_to_fas):
    '''
    path_to_fas: path to fa maps
    '''
    # create subfolder in output dir called 'pngs'
    # to save all of the fa map slices
    outdir = path_to_fas + 'pngs/'
    if (not os.isdir(outdir)): os.mkdir(outdir)
    plt.rcParams.update({'axes.labelsize': 'x-large', 'axes.titlesize':'x-large'})
    def create_subplot(number, title, ylabel=None, xlabel=None, set_ticks=None, set_ticklabels=False):
        ax = plt.subplot(number)
        ax.set_title(title)
        if ylabel != None: ax.set_ylabel(ylabel)
        if xlabel != None: ax.set_xlabel(xlabel)                                                     
        if set_ticks != None: 
            ax.xaxis.set_ticks(set_ticks(0))
            ax.yaxis.set_ticks(set_ticks(1))
        if set_ticklabels:
            ax.get_yaxis().set_ticklabels([])
            ax.get_xaxis().set_ticklabels([])
        return ax
    for f in os.listdir(path_to_fas):
        if (not f.endswith(('.nii', '.nii.gz'))): continue
        img = nib.load(path_to_fas + f)
        data = img.get_data()
        ax_x1 = create_subplot(331, 'X = 78', ylabel='Sagittal Slice: Y and Z fixed', set_ticks=([0, data.shape[1]/2, data.shape[1] - 1 ], [0, data.shape[2]/2, data.shape[2] - 1 ]))
        plt.imshow(ndimage.rotate(data[78,:,:], 90))
        ax_x2 = create_subplot(332, 'X = 90', set_ticklabels=True)  
        plt.imshow(ndimage.rotate(data[90,:,:], 90))
        ax_x3 = create_subplot(333, 'X = 100', set_ticklabels=True)  
        plt.imshow(ndimage.rotate(data[100,:,:], 90))
        ax_y1 = create_subplot(334, 'Y = 82', ylabel='Coronal Slice: X and Z fixed', set_ticks=([0, data.shape[0]/2, data.shape[0] - 1 ], [0, data.shape[2]/2, data.shape[2] - 1 ]))
        plt.imshow(ndimage.rotate(data[:,82,:], 90))
        ax_y2 = create_subplot(335, 'Y = 107', set_ticklabels=True)  
        plt.imshow(ndimage.rotate(data[:,107,:], 90))
        ax_y3 = create_subplot(336, 'Y = 142', set_ticklabels=True)  
        plt.imshow(ndimage.rotate(data[:,142,:], 90))
        ax_z1 = create_subplot(337, 'Z = 88', ylabel='Axial Slice: X and Y fixed', set_ticks=([0, data.shape[0]/2, data.shape[0] - 1 ], [0, data.shape[1]/2, data.shape[1] - 1 ]))
        plt.imshow(data[:,:,88])
        ax_z2 = create_subplot(338, 'Z = 103', set_ticklabels=True)  
        plt.imshow(data[:,:,103])
        ax_z3 = create_subplot(339, 'Z = 107', set_ticklabels=True)  
        plt.imshow(data[:,:,107])
        fig = plt.gcf()
        fig.set_size_inches(12.5, 10.5, forward=True)
        plt.savefig(outdir + f.split(".")[0] + '.png', format='png')
        print(f + " saved!")

def parse_datasets(dir1, dir2):
    files = []
    if dir2 != dir1:
        print('different dirs')
        tensors = [dir1 + '/' + fl
                    for root, dirs, files in os.walk(dir1)
                    for fl in files
                    if fl.endswith('.npz')]
        dtis = [dir2 + '/' + fl
                    for root, dirs, files in os.walk(dir2)
                    for fl in files
                    if fl.endswith(('.nii', '.nii.gz'))]
        print(dtis)
        for ften in tensors:
            fname = (os.path.split(ften)[-1]).split('.')[0]
            print(fname)
            for fdti in dtis:                          
                if (os.path.split(fdti)[-1]).split('.')[0] == fname:
                    print('same files: ' + fname)
                    files.append(tuple(ften, fdti))       
    else:
        for f in os.listdir(dir1):
            if fl.endswith('.npz'): 
                ften = fl
                for f2 in os.listdir(dir1):
                    if (f2.endswith(('.nii', '.nii.gz')) and os.path.splitext(f2)[0] == os.path.splitext(ften)):
                        files.append(tuple(ften, f2))       
    return files 

def main():
    """
    Argument parser and directory crawler. Takes organization and atlas
    information and produces a dictionary of file lists based on datasets
    of interest and then passes it off for processing.
    Required parameters:
        tendir:
            - Basepath for which data can be found directly inwards from
        dtidir:
            - Basepath for which registered dti data can be found directly inwards from
        outdir:
            - Path to derivative save location
    Optional parameters:
        verb:
            - Toggles verbose output statements
    """
    parser = ArgumentParser(description="Computes FA maps and saves FA slices")
    parser.add_argument("tendir", action="store", help="base directory loc")
    parser.add_argument("dtidir", action="store", help="base directory loc")
    parser.add_argument("outdir", action="store", help="base directory loc")
    parser.add_argument("-v", "--verb", action="store_true", help="")
    result = parser.parse_args()

    #  Sets up directory to crawl based on the system organization you're
    #  working on. Which organizations are pretty clear by the code, methinks..
    fs = parse_datasets(result.tendir, result.dtidir)

    if (not os.path.isdir(result.outdir)): os.mkdir(result.outdir)
    #  The fun begins and now we load our tensors and process them.
    tensor2fa(fs, result.outdir)
    save_fa_pngs(result.outdir, result.outdir)

if __name__ == '__main__':
    main()
