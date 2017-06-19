#!/usr/bin/env python

# Copyright 2016 NeuroData (http://neurodata.io)
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

# register.py
# Created by Greg Kiar on 2016-01-28.
# Edited by Eric Bridgeford.
# Email: gkiar@jhu.edu

from subprocess import Popen, PIPE
import os.path as op
import ndmg.utils as mgu
import nibabel as nb
import numpy as np
import nilearn.image as nl
from ndmg.stats.func_qa_utils import registration_score


class register(object):

    def __init__(self):
        """
        Enables registration of single images to one another as well as volumes
        within multi-volume image stacks. Has options to compute transforms,
        apply transforms, as well as a built-in method for aligning low
        resolution dwi images to a high resolution atlas.
        """
        pass

    def align(self, inp, ref, xfm=None, out=None, dof=12, searchrad=True,
              bins=256, interp=None, cost="mutualinfo", sch=None,
              wmseg=None, init=None):
        """
        Aligns two images and stores the transform between them

        **Positional Arguments:**

                inp:
                    - Input impage to be aligned as a nifti image file
                ref:
                    - Image being aligned to as a nifti image file
                xfm:
                    - Returned transform between two images
                out:
                    - determines whether the image will be automatically
                    aligned.
                dof:
                    - the number of degrees of freedom of the alignment.
                searchrad:
                    - a bool indicating whether to use the predefined
                    searchradius parameter (180 degree sweep in x, y, and z).
                interp:
                    - the interpolation method to use.
                sch:
                    - the optional FLIRT schedule file.
                wmseg:
                    - an optional white-matter segmentation for bbr.
                init:
                    - an initial guess of an alignment.
        """
        cmd = "flirt -in {} -ref {}".format(inp, ref)
        if xfm is not None:
            cmd += " -omat {}".format(xfm)
        if out is not None:
            cmd += " -out {}".format(out)
        if dof is not None:
            cmd += " -dof {}".format(dof)
        if bins is not None:
            cmd += " -bins {}".format(bins)
        if interp is not None:
            cmd += " -interp {}".format(interp)
        if cost is not None:
            cmd += " -cost {}".format(cost)
        if searchrad is not None:
            cmd += " -searchrx -180 180 -searchry -180 180 " +\
                   "-searchrz -180 180"
        if sch is not None:
            cmd += " -schedule {}".format(sch)
        if wmseg is not None:
            cmd += " -wmseg {}".format(wmseg)
        if init is not None:
            cmd += " -init {}".format(init)
        mgu.execute_cmd(cmd, verb=True)

    def align_epi(self, epi, t1, brain, out):
        """
        Algins EPI images to T1w image
        """
        cmd = 'epi_reg --epi={} --t1={} --t1brain={} --out={}'
        cmd = cmd.format(epi, t1, brain, out)
        mgu.execute_cmd(cmd, verb=True)

    def align_nonlinear(self, inp, ref, xfm, warp, mask=None):
        """
        Aligns two images using nonlinear methods and stores the
        transform between them.

        **Positional Arguments:**

            inp:
                - the input image.
            ref:
                - the reference image.
            affxfm:
                - the affine transform to use.
            warp:
                - the path to store the nonlinear warp.
            mask:
                - a mask in which voxels will be extracted
                during nonlinear alignment.
        """
        # if we are doing fnirt, use predefined fnirt config file
        # since the config is most robust
        cmd = "fnirt --in={} --aff={} --cout={} --ref={} --config=T1_2_MNI152_2mm"
        cmd = cmd.format(inp, xfm, warp, ref)
        if mask is not None:
            cmd += " --refmask={}".format(mask)
        out, err = mgu.execute_cmd(cmd, verb=True)

    def applyxfm(self, inp, ref, xfm, aligned, interp='trilinear'):
        """
        Aligns two images with a given transform

        **Positional Arguments:**

            inp:
                - Input impage to be aligned as a nifti image file
            ref:
                - Image being aligned to as a nifti image file
            xfm:
                - Transform between two images
            aligned:
                - Aligned output image as a nifti image file
            interp:
                - the interpolation method to use from fsl.
        """
        cmd = "flirt -in {} -ref {} -out {} -init {} -interp {} -applyxfm"
        cmd = cmd.format(inp, ref, aligned, xfm, interp)
        mgu.execute_cmd(cmd, verb=True)

    def apply_warp(self, inp, ref, out, warp=None, xfm=None, mask=None):
        """
        Applies a warp from the functional to reference space
        in a single step, using information about the structural->ref
        mapping as well as the functional to structural mapping.

        **Positional Arguments:**

            inp:
                - the input image to be aligned as a nifti image file.
            out:
                - the output aligned image.
            ref:
                - the image being aligned to.
            warp:
                - the warp from the structural to reference space.
            xfm:
                - the affine transformation to apply to the input.
            mask:
                - the mask to extract around after applying the transform.
        """
        cmd = "applywarp --ref={} --in={} --out={}".format(ref, inp, out)
        if warp is not None:
            cmd += " --warp={}".format(warp)
        if xfm is not None:
            cmd += " --premat={}".format(xfm)
        if mask is not None:
            cmd += " --mask={}".format(mask)
        mgu.execute_cmd(cmd, verb=True)

    def align_slices(self, dwi, corrected_dwi, idx):
        """
        Performs eddy-correction (or self-alignment) of a stack of 3D images

        **Positional Arguments:**
                dwi:
                    - 4D (DTI) image volume as a nifti file
                corrected_dwi:
                    - Corrected and aligned DTI volume in a nifti file
                idx:
                    - Index of the first B0 volume in the stack
        """
        cmd = "eddy_correct {} {} {}".format(dwi, corrected_dwi, idx)
        status = mgu.execute_cmd(cmd, verb=True)

    def resample(self, base, ingested, template):
        """
        Resamples the image such that images which have already been aligned
        in real coordinates also overlap in the image/voxel space.

        **Positional Arguments**
                base:
                    - Image to be aligned
                ingested:
                    - Name of image after alignment
                template:
                    - Image that is the target of the alignment
        """
        # Loads images
        template_im = nb.load(template)
        base_im = nb.load(base)
        # Aligns images
        target_im = nl.resample_img(base_im,
                                    target_affine=template_im.get_affine(),
                                    target_shape=template_im.get_data().shape,
                                    interpolation="nearest")
        # Saves new image
        nb.save(target_im, ingested)

    def resample_fsl(self, base, res, goal_res, interp='nearestneighbour'):
        """
        A function to resample a base image in fsl to that of a template.
        **Positional Arguments:**

            base:
                - the path to the base image to resample.
            res:
                - the filename after resampling.
            goal_res:
                - the desired resolution.
        """
        cmd = "flirt -in {} -ref {} -out {} -applyisoxfm {} -interp {}"
        cmd = cmd.format(base, base, res, goal_res, interp)
        mgu.execute_cmd(cmd, verb=True)

    def combine_xfms(self, xfm1, xfm2, xfmout):
        """
        A function to combine two transformations, and output the
        resulting transformation.

        **Positional Arguments**
            xfm1:
                - the path to the first transformation
            xfm2:
                - the path to the second transformation
            xfmout:
                - the path to the output transformation
        """
        cmd = "convert_xfm -omat {} -concat {} {}".format(xfmout, xfm1, xfm2)
        mgu.execute_cmd(cmd, verb=True)

    def dwi2atlas(self, dwi, gtab, t1w, atlas,
                  aligned_dwi, outdir, clean=False):
        """
        Aligns two images and stores the transform between them

        **Positional Arguments:**

                dwi:
                    - Input impage to be aligned as a nifti image file
                gtab:
                    - object containing gradient directions and strength
                t1w:
                    - Intermediate image being aligned to as a nifti image file
                atlas:
                    - Terminal image being aligned to as a nifti image file
                aligned_dwi:
                    - Aligned output dwi image as a nifti image file
                outdir:
                    - Directory for derivatives to be stored
        """
        # Creates names for all intermediate files used
        dwi_name = mgu.get_filename(dwi)
        t1w_name = mgu.get_filename(t1w)
        atlas_name = mgu.get_filename(atlas)

        dwi2 = mgu.name_tmps(outdir, dwi_name, "_t2.nii.gz")
        temp_aligned = mgu.name_tmps(outdir, dwi_name, "_ta.nii.gz")
        temp_aligned2 = mgu.name_tmps(outdir, dwi_name, "_ta2.nii.gz")
        b0 = mgu.name_tmps(outdir, dwi_name, "_b0.nii.gz")
        t1w_brain = mgu.name_tmps(outdir, t1w_name, "_ss.nii.gz")
        xfm = mgu.name_tmps(outdir, t1w_name,
                            "_" + atlas_name + "_xfm.mat")

        # Align DTI volumes to each other
        self.align_slices(dwi, dwi2, np.where(gtab.b0s_mask)[0][0])

        # Loads DTI image in as data and extracts B0 volume
        dwi_im = nb.load(dwi2)
        b0_im = mgu.get_b0(gtab, dwi_im.get_data())

        # Wraps B0 volume in new nifti image
        b0_head = dwi_im.get_header()
        b0_head.set_data_shape(b0_head.get_data_shape()[0:3])
        b0_out = nb.Nifti1Image(b0_im, affine=dwi_im.get_affine(),
                                header=b0_head)
        b0_out.update_header()
        nb.save(b0_out, b0)

        # Applies skull stripping to T1 volume, then EPI alignment to T1
        mgu.extract_brain(t1w, t1w_brain, ' -B')
        self.align_epi(dwi2, t1w, t1w_brain, temp_aligned)

        # Applies linear registration from T1 to template
        self.align(t1w, atlas, xfm)

        # Applies combined transform to dwi image volume
        self.applyxfm(temp_aligned, atlas, xfm, temp_aligned2)
        self.resample(temp_aligned2, aligned_dwi, atlas)

        if clean:
            cmd = "rm -f {} {} {} {} {}*".format(dwi2, temp_aligned, b0,
                                                    xfm, t1w_name)
            print("Cleaning temporary registration files...")
            mgu.execute_cmd(cmd)


class func_register(register):
    def __init__(self, func, t1w, atlas, atlas_brain, atlas_mask,
                 aligned_func, aligned_t1w, outdir):
        """
        A class to change brain spaces from a subject's epi sequence
        to that of a standardized atlas.

        **Positional Arguments:**

            func:
                - the path of the preprocessed fmri image.
            t1w:
                - the path of the T1 scan.
            atlas:
                - the template atlas.
            atlas_brain:
                - the template brain.
            atlas_mask:
                - the template mask.
            aligned_func:
                - the name of the aligned fmri scan to produce.
            aligned_t1w:
                - the name of the aligned anatomical scan to produce
            outdir:
                - the output base directory.
        """
        super(register, self).__init__()
        # our basic dependencies
        self.epi = func
        self.atlas = atlas
        self.atlas_brain = atlas_brain
        self.atlas_mask = atlas_mask
        self.taligned_epi = aligned_func
        self.taligned_t1w = aligned_t1w
        self.outdir = outdir
        # paths to intermediates for qa later
        self.sreg_strat = None
        self.treg_strat = None

        # for naming temporary files
        self.epi_name = mgu.get_filename(func)
        self.t1w_name = mgu.get_filename(t1w)
        self.atlas_name = mgu.get_filename(atlas)

        # put anatomical in 2mm resolution for memory
        # efficiency if it is lower
        self.simp=False  # for simple inputs
        if sum(nb.load(t1w).header.get_zooms()) < 9:
            self.t1w = "{}/{}_resamp.nii.gz".format(self.outdir['sreg_a'],
                                                    self.t1w_name)
            self.resample_fsl(t1w, self.t1w, 2)
        else:
            self.simp = True  # if the input is poor
            self.t1w = t1w
        # since we will need the t1w brain multiple times
        self.t1w_brain = "{}/{}_brain.nii.gz".format(self.outdir['sreg_a'],
                                                     self.t1w_name)
        # Applies skull stripping to T1 volume
        # using a very low sensitivity for thresholding
        self.t1_bet_sens = '-f 0.3 -R -B -S'
        self.fm_bet_sens = '-f 0.3 -R'
        mgu.extract_brain(self.t1w, self.t1w_brain, opts=self.t1_bet_sens)
        # name intermediates for self-alignment
        self.saligned_xfm = "{}/{}_self-aligned.mat".format(
            self.outdir['sreg_f'],
            self.epi_name)
        pass

    def self_align(self):
        """
        A function to perform self alignment. Uses a local optimisation
        cost function to get the two images close, and then uses bbr
        to obtain a good alignment of brain boundaries.
        """
        xfm_init1 = "{}/{}_xfm_epi2t1w_init1.mat".format(self.outdir['sreg_f'],
                                                         self.epi_name)
        xfm_init2 = "{}/{}_xfm_epi2t1w_init2.mat".format(self.outdir['sreg_f'],
                                                         self.epi_name)
        epi_init = "{}/{}_local.nii.gz".format(self.outdir['sreg_f'],
                                               self.epi_name)

        # perform an initial alignment with a gentle translational guess
        self.align(self.epi, self.t1w_brain, xfm=xfm_init1, bins=None,
                   dof=None, cost=None, searchrad=None,
                   sch="${FSLDIR}/etc/flirtsch/sch3Dtrans_3dof")
        # perform a local registration
        self.align(self.epi, self.t1w_brain, xfm=xfm_init2, init=xfm_init1,
                   bins=None, dof=None, cost=None, searchrad=None,
                   out=epi_init, sch="${FSLDIR}/etc/flirtsch/simple3D.sch")

        # attempt EPI registration. note that this sometimes does not
        # work great if our EPI has a low field of view.
        if not self.simp:
            xfm_init3 = "{}/{}_xfm_epi2t1w.mat".format(self.outdir['sreg_f'],
                                                       self.epi_name) 
            xfm_bbr = "{}/{}_xfm_bbr.mat".format(self.outdir['sreg_f'],
                                                 self.epi_name)
            epi_bbr = "{}/{}_bbr.nii.gz".format(self.outdir['sreg_f'],
                                                self.epi_name)
            self.align(self.epi, self.t1w_brain, xfm=xfm_init3,
                       init=xfm_init2, bins=None, dof=6, cost=None,
                       searchrad=None, sch=None)
            map_path = "{}/{}_t1w_seg".format(self.outdir['sreg_f'],
                                              self.t1w_name)
            maps = mgu.segment_anat(self.t1w_brain, map_path)
            wm_mask = "{}/{}_wmm.nii.gz".format(self.outdir['sreg_f'],
                                                self.t1w_name)
            mgu.extract_mask(maps['wm_prob'], wm_mask, 0.5)
            self.align(self.epi, self.t1w, xfm=xfm_bbr, wmseg=wm_mask,
                       out=epi_bbr, init=xfm_init3, interp="spline",
                       sch="${FSLDIR}/etc/flirtsch/bbr.sch")
            self.sreg_xfm = xfm_bbr
            self.sreg_brain = epi_bbr
            self.sreg_strat = 'epireg'
        else:
            print ("Warning: BBR self registration not "
                   "attempted, as input is low quality.")
            self.sreg_xfm = xfm_init2
            self.sreg_brain = epi_init
            self.sreg_strat = 'flirt'
        mgu.extract_brain(self.sreg_brain, self.sreg_brain,
                          opts=self.fm_bet_sens)
        pass

    def template_align(self):
        """
        A function to perform template alignment. First tries nonlinear
        registration, and if that does not work effectively, does a linear
        registration instead.
        NOTE: for this to work, must first have called self-align.
        """
         
        xfm_t1w2temp = "{}/{}_xfm_t1w2temp.mat".format(self.outdir['treg_a'],
            self.epi_name)

        # linear registration from t1 space to atlas space
        self.align(self.t1w_brain, self.atlas_brain, xfm=xfm_t1w2temp,
                   out=None, dof=12, searchrad=True, bins=256, interp="spline",
                   wmseg=None, init=None)

        # if the atlas is MNI 2mm, then we have a config file for it
        if (nb.load(self.atlas).get_data().shape in [(91, 109, 91)] and
            (self.simp is False)):
            warp_t1w2temp = "{}/{}_warp_t1w2temp.nii.gz".format(
                self.outdir['treg_a'],
                self.epi_name
            )
            #epi_nl = "{}/{}_temp-aligned_nonlinear.nii.gz".format(
            #    self.outdir['treg_f'],
            #    self.epi_name)
            #t1w_nl = "{}/{}_temp-aligned_nonlinear.nii.gz".format(
            #    self.outdir['treg_a'],
            #    self.t1w_name)
            self.align_nonlinear(self.t1w, self.atlas, xfm_t1w2temp,
                                 warp_t1w2temp, mask=self.atlas_mask)
            self.apply_warp(self.epi, self.atlas, self.taligned_epi,
                            warp=warp_t1w2temp, xfm=self.sreg_xfm)
            self.apply_warp(self.t1w, self.atlas, self.taligned_t1w,
                            warp=warp_t1w2temp)
            self.treg_strat = 'fnirt'
        else:
            print "Atlas is not 2mm MNI, or input is low quality."
            print "Using linear template registration."

            # epi_lin = "{}/{}_temp-aligned_linear.nii.gz".format(
            #     self.outdir['treg_f'],
            #     self.epi_name)
            # t1w_lin = "{}/{}_temp-aligned_linear.nii.gz".format(
            #     self.outdir['treg_a']
            #     self.epi_name)
            xfm_epi2temp = "{}/{}_xfm_epi2temp.mat".format(
                self.outdir['treg_f'],
                self.epi_name
            )
            # just apply our previously computed linear transform
            self.combine_xfms(xfm_t1w2temp, self.sreg_xfm, xfm_epi2temp)
            self.applyxfm(self.epi, self.atlas, xfm_epi2temp,
                          self.taligned_epi, interp='spline')
            self.apply_warp(self.t1w, self.atlas, self.taligned_t1w,
                            xfm=xfm_t1w2temp) 
            self.treg_strat = 'flirt'
        self.taligned_epi_mask = "{}/{}_temp-aligned_mask.nii.gz".format(
            self.outdir['treg_f'],
            self.epi_name
        )
        mgu.extract_brain(self.taligned_epi, self.taligned_epi_mask,
                          opts=self.fm_bet_sens + ' -m')
        mgu.apply_mask(self.taligned_epi, self.taligned_epi,
                       self.taligned_epi_mask)
        mgu.extract_brain(self.taligned_t1w, self.taligned_t1w,
                          opts=self.t1_bet_sens)
        pass

    def register(self):
        """
        A function to perform self registration followed by
        template registration.
        """
        self.self_align()
        self.template_align()
        pass
