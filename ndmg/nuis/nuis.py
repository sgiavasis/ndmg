#!/usr/bin/env python

# Copyright 2016 NeuroData (http://neuromri_dat.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# nuis.py
# Created by Eric Bridgeford on 2016-06-20-16.
# Email: ebridge2@jhu.edu
from ndmg.utils import utils as mgu
import nibabel as nb
import numpy as np
from scipy.fftpack import rfft, irfft, rfftfreq


class nuis(object):

    def __init__(self, fmri, smri, nuis_mri, outdir, lv_mask=None):
        """
        A class for nuisance correction of fMRI.

        **Positional Arguments:**

            - fmri:
                - the functional mri.
            - smri:
                - the structural mri (assumed to be T1w).
            - nuis_mri:
                - the file path of a nuisance corrected mri
            - lv_mask:
                - lateral-ventricles mask (optional).
        """
        # store our inputs
        self.fmri = fmri  # the fmri
        self.smri = smri  # the T1w anatomical mri
        self.nuis_mri = nuis_mri  # the nuisance-corrected path

        # store the masks so that we can easily do qa later
        self.lv_mask = lv_mask  # lateral-ventricles mask
        self.er_wm_mask = None  # eroded white-matter mask

        # places to place intermediates
        self.outdir = outdir
        self.anat_name = mgu.get_filename(smri)
        # store temporary path for segmentation step
        nuisname = "{}_nuis".format(self.anat_name)
        self.nuisname = nuisname
        # wm mask
        self.wm_mask = mgu.name_tmps(outdir, nuisname, "_wmm.nii.gz")
        # csf mask not used due to inconsistencies in segmenting
        # gm mask
        self.gm_mask = mgu.name_tmps(outdir, nuisname, "_gmm.nii.gz")
        self.map_path = mgu.name_tmps(outdir, nuisname, "_seg")
        # segment the brain for quality control purposes
        self.segment_anat(self.smri, self.map_path)
        # extract the masks
        self.extract_mask(self.wm_prob, self.wm_mask, .99, erode=0)
        self.extract_mask(self.gm_prob, self.gm_mask, .95, erode=0)
        # the centered brain
        self.cent_nuis = None
        # the brain after glm
        self.glm_nuis = None
        # the brain after frequency correction
        self.fft_nuis = None
        # regressors so that we can use them for line plots
        self.csf_reg = None
        self.wm_reg = None
        self.quad_reg = None
        self.fft_reg = None
        self.fft_bef = None

        # signal that is removed at given steps
        self.fft_sig = None
        self.glm_sig = None
        pass

    def center_signal(self, data):
        """
        A function that performs normalization to a
        given fMRI signal, non-transposed. We subtract out the mean of
        each dimension.

        **Positional Arguments:**

            data:
                - the fMRI data. Should be passed as an ndarray,
                  with dimensions [xdim, ydim, zdim, ntimesteps].
        """
        print "Centering Signal..."
        data = data - data.mean(axis=3, keepdims=True)
        return data

    def normalize_signal(self, data):
        """
        A function taht performs normalization by the standard deviation
        to a fMRI signal. We divide by the standard deviation of the signal
        so that the voxel timeseries are of relatively equal magnitude,
        as the standard deviation is the primary contrast that we are
        concerned with.

        **Positional Arguments:**

            data:
                - the fMRI data. Should be an array, with dimensions
                  [ntimesteps, nvoxels].
        """
        print "Normalizing Signal by Standard Deviation..."
        # normalize the signal
        data = np.divide(data, data.std(axis=0))
        # returns the normalized signal
        return data

    def compcor(self, masked_ts, n=5):
        """
        A function to extract principal components on
        timeseries of nuisance variables.

        **Positional Arguments:**

            masked_ts:
                - the timeseries over a masked region. We assume
                  that this array is already mean centered per voxel.
                  Dimensions should be [ntimesteps, nvoxels].
           n:
                - the number of components to use.
        """
        print "Extracting Nuisance Components..."
        # singular value decomposition to get the ordered
        # principal components
        U, s, V = np.linalg.svd(masked_ts)
        # return the top n principal components
        return U[:, 0:n], s
        pass

    def freq_filter(self, mri, tr, highpass=0.01, lowpass=None):
        """
        A function that uses scipy's fft and ifft to frequency filter
        an fMRI image.

        **Positional Arguments:**
            mri:
                - an ndarray containing timeseries of dimensions
                  [voxels,timesteps] which the user wants to have
                  frequency filtered.
            highpass:
                - the lower limit frequency band to remove below.
            lowpass:
                - the upper limit  frequency band to remove above.
        """
        # apply the fft per voxel to take to fourier domain
        passed_fft = np.apply_along_axis(rfft, 0, mri)

        # get the frequencies returned by the fft that we want
        # to use. note that this function returns us a single-sided
        # set of frequencies
        freq_ra = rfftfreq(mri.shape[0], d=tr)
        self.freq_ra = np.sort(freq_ra)
        order = np.argsort(freq_ra)

        # free for memory purposes
        mri = None
        
        self.fft_bef = np.square(passed_fft[:,
            self.voxel_gm_mask].mean(axis=1))[order]
        bpra = np.zeros(freq_ra.shape, dtype=bool)
        # figure out which positions we will exclude
        if highpass is not None:
            print "filtering below " + str(highpass) + " Hz..."
            bpra[np.abs(freq_ra) < highpass] = True
        if lowpass is not None:
            print "filtering above " + str(lowpass) + " Hz..."
            bpra[np.abs(freq_ra) > lowpass] = True
        print "Applying Frequency Filtering..."
        filtered_ra = np.logical_not(bpra)
        filtered_fft = passed_fft.copy()
        filtered_fft[filtered_ra, :] = 0
        filt_sig = np.apply_along_axis(irfft, 0,
                                       filtered_fft)
        filtered_fft = None
        self.fft_sig = filt_sig[:, self.voxel_gm_mask].mean(axis=1)
        filt_sig = None

        passed_fft[bpra, :] = 0
        self.fft_reg = np.square(passed_fft[:, 
                self.voxel_gm_mask].mean(axis=1))[order]
        # go back to time domain
        return np.apply_along_axis(irfft, 0, passed_fft)

    def regress_signal(self, data, R):
        """
        Regresses data to given regressors.

        **Positional Arguments:**
            - data:
                - the data as a ndarray.
            - R:
                - a numpy ndarray of regressors to
                  regress to.
        """
        print "GLM with Design Matrix of Dimensions " + str(R.shape) + "..."
        # OLS solution for GLM B = (X^TX)^(-1)X^TY
        coefs = np.linalg.inv(R.T.dot(R)).dot(R.T).dot(data)
        return R.dot(coefs)

    def segment_anat(self, amri, basename, an=1):
        """
        A function to use FSL's FAST to segment an anatomical
        image into GM, WM, and CSF maps.

        **Positional Arguments:**

            - amri:
                - an anatomical image.
            - basename:
                - the basename for outputs. Often it will be
                  most convenient for this to be the dataset,
                  followed by the subject, followed by the step of
                  processing. Note that this anticipates a path as well;
                  ie, /path/to/dataset_sub_nuis, with no extension.
            - an:
                - an integer representing the type of the anatomical image.
                  (1 for T1w, 2 for T2w, 3 for PD).
        """
        print "Segmenting Anatomical Image into WM, GM, and CSF..."
        # run FAST, with options -t for the image type and -n to
        # segment into CSF (pve_0), WM (pve_1), GM (pve_2)
        cmd = " ".join(["fast -t", str(int(an)), "-n 3 -o", basename, amri])
        mgu.execute_cmd(cmd)

        self.wm_prob = "{}_{}".format(basename, "pve_2.nii.gz")
        self.gm_prob = "{}_{}".format(basename, "pve_1.nii.gz")
        pass

    def erode_mask(self, mask, v=0):
        """
        A function to erode a mask by a specified number of
        voxels. Here, we define erosion as the process of checking
        whether all the voxels within a number of voxels for a
        mask have values.

        **Positional Arguments:**

            - mask:
                - a numpy array of a mask to be eroded.
            - v:
                - the number of voxels to erode by.
        """
        print "Eroding Mask..."
        for i in range(0, v):
            # masked_vox is a tuple 0f [x]. [y]. [z] cooords
            # wherever mask is nonzero
            erode_mask = np.zeros(mask.shape)
            x, y, z = np.where(mask != 0)
            if (x.shape == y.shape and y.shape == z.shape):
                # iterated over all the nonzero voxels
                for j in range(0, x.shape[0]):
                    # check that the 3d voxels within 1 voxel are 1
                    # if so, add to the new mask
                    md = mask.shape
                    if (mask[x[j], y[j], z[j]] and
                            mask[np.min((x[j]+1, md[0]-1)), y[j], z[j]] and
                            mask[x[j], np.min((y[j]+1, md[1]-1)), z[j]] and
                            mask[x[j], y[j], np.min((z[j]+1, md[2]-1))] and
                            mask[np.max((x[j]-1, 0)), y[j], z[j]] and
                            mask[x[j], np.max((y[j]-1, 0)), z[j]] and
                            mask[x[j], y[j], np.max((z[j]-1, 0))]):
                        erode_mask[x[j], y[j], z[j]] = 1
            else:
                raise ValueError('Your mask erosion has an invalid shape.')
            mask = erode_mask
        return mask

    def extract_mask(self, prob_map, mask_path, t, erode=0):
        """
        A function to extract a mask from a probability map.
        Also, performs mask erosion as a substep.

        **Positional Arguments:**

            - prob_map:
                - the path to probability map for the given class
                  of brain tissue.
            - mask_path:
                - the path to the extracted mask.
            - t:
                - the threshold to consider voxels part of the class.
            - erode=2:
                - the number of voxels to erode by. Defaults to 2.
        """
        print "Extracting Mask from probability map {}...".format(prob_map)
        prob = nb.load(prob_map)
        prob_dat = prob.get_data()
        mask = (prob_dat > t).astype(int)
        mask = self.erode_mask(mask, v=erode)
        img = nb.Nifti1Image(mask,
                             header=prob.header,
                             affine=prob.get_affine())
        # save the corrected image
        nb.save(img, mask_path)
        return mask_path

    def linear_reg(self, voxel, csf_ts=None, wm_ts=None, n=None):
        """ 
        A function to perform quadratic detrending of fMRI data.

        **Positional Arguments**

            - voxel:
                - an ndarray containing a voxel timeseries.
                  dimensions should be [timesteps, voxels]
            - csf_ts:
                - a timeseries for csf mean regression. If not
                  provided, csf regression will not be performed.
            - wm_ts:
                - a timeseries for white matter regression.
                  If only wm_ts is provided, wm mean regression
                  will be performed. If n and wm_ts are provided, 
                  compcor with n components will be performed. If
                  neither are provided, no wm regression will be
                  performed.
            - n:
                - the number of components for wm regression.
        """
        # time dimension is now the 0th dim
        time = voxel.shape[0]
        # linear drift regressor
        lin_reg = np.array(range(0, time))
        # quadratic drift regressor
        quad_reg = np.array(range(0, time))**2

        # use GLM model given regressors to approximate the weight we want
        # to regress out
        R = np.column_stack((np.ones(time), lin_reg, quad_reg))

        # highpass filter voxel timeseries appropriately

        if csf_ts is not None:
            csf_reg = csf_ts.mean(axis=1, keepdims=True)
            self.csf_reg = csf_reg  # save for qa later
            # add coefficients to our regression
            R = np.column_stack((R, csf_reg))

        if n is not None and wm_ts is not None:
            wm_reg = self.compcor(wm_ts, n=n)
            self.wm_reg = wm_reg  # save for QA later
            R = np.column_stack((R, wm_reg))
        elif wm_ts is not None:
            wm_reg = wm_ts.mean(axis=1, keepdims=True)
            self.wm_reg = wm_reg
            R = np.column_stack((R, wm_reg[:, 0]))

        W = self.regress_signal(voxel, R)
        self.glm_sig = W[:, self.voxel_gm_mask].mean(axis=1)

        # corr'd ts is the difference btwn the original timeseries and
        # our regressors, and then we transpose back
        return (voxel - W)

    def nuis_correct(self, highpass=0.01, lowpass=None, trim=0, n=None):
        """
        Removes Nuisance Signals from brain images, using a combination
        of Frequency filtering, and mean csf/quadratic regression.

        **Positional Arguments:**

            - highpass:
                - the highpass cutoff for FFT.
            - lowpass:
                - the lowpass cutoff for FFT. NOT recommended.
            - trim:
                - trim the timeseries by a number of slices. Corrects
                  for T1 effects; that is, in some datasets, the first few
                  timesteps may have a non-saturated T1 contrast and as such
                  will show non-standard intensities.
            - n:
                - the number of components for wm regression. If set to None,
                  does not perform wm regression.
        """
        fmri_name = mgu.get_filename(self.fmri)
        fmri_im = nb.load(self.fmri)

        fmri_dat = fmri_im.get_data()
        basic_mask = fmri_dat.sum(axis=3) > 0
        gm_mask_dat = nb.load(self.gm_mask).get_data()
        # mean center signal to start with
        fmri_dat = self.center_signal(fmri_dat)

        # load the voxel timeseries and transpose
        # remove voxels that are absolutely non brain (zero activity)
        # and trim here so that we don't correct for nuis timepoints
        voxel = fmri_dat[basic_mask, :].T

        # zooms are x, y, z, t
        tr = fmri_im.header.get_zooms()[3]

        if self.lv_mask is not None:
            # csf regressor is the mean of all voxels in the csf
            # mask at each time point
            lvm = nb.load(self.lv_mask).get_data()
            lv_ts = fmri_dat[lvm != 0, :].T
        else:
            lv_ts = None

        # if n is provided, perform compcor
        if n is not None:
            self.er_wm_mask = '{}_{}.nii.gz'.format(self.map_path,
                                                 "wm_mask_eroded")
            self.extract_mask(self.wm_prob, self.er_wm_mask, .99, erode=2)
            wmm = nb.load(self.er_wm_mask).get_data()
            wm_ts = fmri_dat[wmm != 0, :].T
        else:
            wm_ts = None

        fmri_dat = None  # free for memory purposes
        self.voxel_gm_mask = gm_mask_dat[basic_mask == True] > 0
        gm_mask_dat = None  # free for memory
        self.cent_nuis = voxel[:, self.voxel_gm_mask].mean(axis=1)
        # GLM for nuisance correction
        voxel = self.linear_reg(voxel, csf_ts=lv_ts,
                                wm_ts=wm_ts, n=n)
        self.glm_nuis = voxel[:, self.voxel_gm_mask].mean(axis=1)

        # Frequency Filtering for Nuisance Correction
        voxel = self.freq_filter(voxel, tr, highpass=highpass,
                                 lowpass=lowpass)
        self.fft_nuis = voxel[:, self.voxel_gm_mask].mean(axis=1)

        # normalize the signal to account for anatomical intensity differences
	# self.voxel = self.normalize_signal(self.voxel)
        # put the nifti back together again and re-transpose
        fmri_dat = fmri_im.get_data()
        fmri_dat[basic_mask, :] = voxel.T

        # free for memory purposes
        voxel = None
        fmri_dat = fmri_dat[:, :, :, trim:]
        img = nb.Nifti1Image(fmri_dat,
                             header=fmri_im.header,
                             affine=fmri_im.affine)
        # save the corrected image
        nb.save(img, self.nuis_mri)
        pass
