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

# preproc_fmri.py
# Created by Eric Bridgeford on 2016-06-20-16.
# Email: ebridge2@jhu.edu

import numpy as np
import nibabel as nb
import sys
import os.path as op
import os.path
import nilearn as nl
from ndmg.utils import utils as mgu
from scipy import signal


class preproc_func():

    def __init__(self, func, preproc_func, motion_func,
                   outdir, stc=None, scanid=""):
        """
        Enables preprocessing of single images for single images. Has options
        to perform motion correction.
        """
        self.func = func
        self.preproc_func = preproc_func
        self.motion_func = motion_func
        self.outdir = outdir
        self.stc = stc
        self.scanid = scanid
        pass

    def motion_correct(self, mri, corrected_mri, idx=None):
        """
        Performs motion correction of a stack of 3D images.

        **Positional Arguments:**
            mri (String):
                 -the 4d (fMRI) image volume as a nifti file.
            corrected_mri (String):
                - the corrected and aligned fMRI image volume.
            idx:
                - the index to use as a reference for self
                  alignment. Uses the meanvolume if not specified
        """
        if idx is None:
            cmd = "mcflirt -in {} -out {} -plots -meanvol"
            cmd = cmd.format(mri, corrected_mri)
        else:
            cmd = "mcflirt -in {} -out {} -plots -refvol {}"
            cmd = cmd.format(mri, corrected_mri, idx)
        mgu.execute_cmd(cmd, verb=True)

    def slice_time_correct(self, func, corrected_func, stc=None):
        """
        Performs slice timing correction of a stack of 3D images.

        **Positional Arguments:**
            mri (String):
                 -the 4d (fMRI) image volume as a nifti file.
            corrected_mri (String):
                - the corrected and aligned fMRI image volume.
            stc: the slice timing correction options, a string
                  corresponding to the acquisition sequence.
                  Options are "/path/to/file", "down", "up",
                  "interleaved". If a file is passed, each line
                  should correspond to a single value (in TRs)
                  of the shift of each slice. For example,
                  if the first slice has no shift, the first line
                  in the text file would be "0.5".
                  If not None, make sure the "zooms" property is set
                  in your data (nb.load(mri).header.get_zooms()), otherwise
                  this function will throw an error.
        """
        if (stc is not None):
            cmd = "slicetimer -i {} -o {}".format(func, corrected_func)
            if stc == "down":
                cmd += " --down"
            elif stc == "interleaved":
                cmd += " --odd"
            elif stc == "up":
                cmd += '' # default
            elif op.isfile(stc):
                cmd += " --tcustom {}".format(stc)
            zooms = nb.load(func).header.get_zooms()
            cmd += " -r {}".format(zooms[3])
            mgu.execute_cmd(cmd, verb=True)
        else:
            print "Skipping slice timing correction."

    def preprocess(self):
        """
        A function to preprocess a stack of 3D images.
        """
        func_name = mgu.get_filename(func)

        s0 = "{}/{}_0slice.nii.gz".format(outdir, func_name)
        stc_func = "{}/{}_stc.nii.gz".format(outdir, func_name)
        # TODO EB: decide whether it is advantageous to align to mean image
        if (stc is not None):
            self.slice_time_correct(self.func, stc_func, stc)
        else:
            stc_func = self.func
        self.motion_correct(stc_func, self.motion_func, None)
        self.mc_params = "{}.par".format(self.motion_func)
        cmd = "cp {} {}".format(self.motion_func, self.preproc_func)
        mgu.execute_cmd(cmd, verb=True)
