from dipy.viz import window, actor
import numpy as np
import nibabel as nib
import random
import sys
import os
import re
import vtk
import paramiko
import getpass
import subprocess

def visualize(fibers, atlas_volume, opacity, outf=None):
    """
    Takes fiber streamlines and visualizes them using DiPy
    Required Arguments:
        - fibers:
            fiber streamlines in a list as returned by DiPy
    Optional Arguments:
        - save:
            flag indicating whether or not you want the image saved
            to disk after being displayed
    """
    
    # Initialize renderer
    renderer = window.Renderer()

    # Add streamlines as a DiPy viz object
    stream_actor = actor.line(fibers)

    # Set camera orientation properties
    # TODO: allow this as an argument
    renderer.set_camera()  # args are: position=(), focal_point=(), view_up=()

    # Add streamlines to viz session
    renderer.add(stream_actor)
    renderer.add(atlas_volume)
    
    # Display fibers
    # TODO: allow size of window as an argument
    # window.show(renderer, size=(600, 600), reset_camera=False)

    # Saves file, if you're into that sort of thing...
    if outf is not None:
        window.record(renderer, out_path=outf, size=(600, 600))
        print('done')   

def threshold_fibers(fibs):
    fib_lengths = [len(f) for f in fibs]
    med = np.median(fib_lengths)
    maximum = max(fib_lengths)
    minimum = min(fib_lengths)
    long_fibs = [f for f in fibs if len(f) > med]
    return long_fibs

def random_sample(fibs, num_samples):
    samples = random.sample(range(len(fibs)), num_samples)
    return [fibs[i] for i in samples]

def load_atlas(path):
    nifti_reader = vtk.vtkNIFTIImageReader()
    nifti_reader.SetFileName(path)
    nifti_reader.Update()

    # The following class is used to store transparencyv-values for later retrival. In our case, we want the value 0 to be
    # completly opaque whereas the three different cubes are given different transperancy-values to show how it works.
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(1, 0.04)

    # This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
    # to be of the colors red green and blue.
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(1, 1.0, 1.0, 1.0)

    # The preavius two classes stored properties. Because we want to apply these properties to the volume we want to render,
    # we have to store them in a class that stores volume prpoperties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    volumeProperty.ShadeOn()


    # This class describes how the volume is rendered (through ray tracing).
    # compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    # We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
    volumeMapper = vtk.vtkSmartVolumeMapper()
    # volumeMapper.SetBlendModeToComposite()
    volumeMapper.SetInputDataObject(nifti_reader.GetOutput())

    # The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return volume

def main(fibers, num_samples, atlas_volume, opacity, outf=None):
    fibs = np.load(fibers)
    fibs = fibs[fibs.keys()[0]]
    fibs = threshold_fibers(fibs)
    resampled_fibs = random_sample(fibs, num_samples)

    
    if outf is not None:
        visualize(resampled_fibs, atlas_volume, opacity, outf)
    else:
        visualize(resampled_fibs, atlas_volume, opacity)

if __name__ == "__main__":
    # get ssh password (only once)
    pw = getpass.getpass()
    # end get password
    # set up SSH stuff
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('braincloud1.cs.jhu.edu', username='vikram', password=pw)
    sftp = ssh.open_sftp()
    # end set up SSH stuff
    datasets = ['KKI2009', 'SWU4', 'BNU1', 'HNU1', 'NKI1', 'MRN1313', 'MRN114']
    num_fibs = 10000
    fiberdir = '/Users/vikramchandrashekhar/Desktop/Grad_School/jovo_rotation/fiber_pngs/'
    opacity = 0.02
    base_path = '/brainstore/MR/data/'
    fib_suffix = '/ndmg_v0033/fibers/'
    fname_pat = re.compile('(^[A-Z]+\d+)_([^_]+)_([^_]+)')
    atlas_volume = load_atlas('atlases/MNI152_T1_1mm_brain_mask.nii.gz')
    for d in datasets:
        path_to_fibers = base_path + d + fib_suffix
        print(path_to_fibers)
        for f in sftp.listdir(path_to_fibers):
            if (not re.match('^.*npz', f )): continue
            result = fname_pat.match(f)
            dataset = result.group(1)
            subject = result.group(2)
            trial = result.group(3)
            subj_trial = subject + '_' + trial
            old = False
            # check if qc already performed for this data
            for f2 in os.listdir(fiberdir + d):
                if (f2.split('.')[-1] != 'png'): continue
                if (f2.startswith(f.split('.')[0])):
                    os.system('echo ' + f2 + ' performed')
                    old = True
                    break
            if (old): continue
            print(f + ' working')
            fname = base_path + d + fib_suffix  + f
            fname_local = fiberdir + d + '/' + f
            x = subprocess.check_call(['rsync', '--append', '--progress', '--rsh=sshpass -p ' +  pw + ' ssh -l vikram', 'braincloud1.cs.jhu.edu:' + fname,  fiberdir + d])
            print(x)
            main(fname_local, num_fibs, atlas_volume, opacity, outf=(fname_local + '.png'))
            os.system('rm ' + fname_local)
