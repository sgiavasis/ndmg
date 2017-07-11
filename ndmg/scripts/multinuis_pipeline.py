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

# ndmg_func_pipeline.py
# Created by Eric Bridgeford on 2016-06-07.
# Edited by Greg Kiar on 2017-03-14.
# Email: gkiar@jhu.edu, ebridge2@jhu.edu

import numpy as np
import nibabel as nb
import os.path as op
from argparse import ArgumentParser
from datetime import datetime
from ndmg.utils import utils as mgu
from ndmg import epi_register as mgreg
from ndmg import graph as mgg
from ndmg.timeseries import timeseries as mgts
from ndmg.stats.qa_func import qa_func
from ndmg.preproc import preproc_func as mgfp
from ndmg.preproc import preproc_anat as mgap
from ndmg.nuis import nuis as mgn
from ndmg.stats.qa_reg import *
import traceback


def multinuis_pipeline(func, t1w, lv_mask, labels, outdir, clean=False, fmt='gpickle',
                       mc_params=None, highpass=.01, lowpass=None, cc=5, csf_mean=False,
                       wm_mean=False, mot=24, trend='quad'):
    """
    analyzes fmri images and produces subject-specific derivatives.

    **positional arguments:**
        func:
            - the path to a 4d (fmri) image.
        t1w:
            - the path to a 3d (anatomical) image.
        lv_mask:
            - the path to the lateral ventricles mask.
        mc_params:
            - the motion parameters file.
        labels:
            - a list of labels files.
        outdir:
            - the base output directory to place outputs.
        clean:
            - a flag whether or not to clean out directories once finished.
        fmt:
            - the format for produced connectomes. supported options are
              gpickle and graphml.
    """
    startTime = datetime.now()

    # Create derivative output directories
    func_name = mgu.get_filename(func)
    t1w_name = mgu.get_filename(t1w)

    paths = {'nuis': "nuis",
             'ts_voxel': "timeseries/voxel",
             'ts_roi': "timeseries/roi"}
    finals = {'ts_roi': paths['ts_roi'],
              'ts_voxel': paths['ts_voxel'],
              'conn': "connectomes"}

    tmpdir = '{}/tmp/{}'.format(outdir, func_name)
    qadir = "{}/qa/{}".format(outdir, func_name)

    tmp_dirs = {}
    qa_dirs = {}
    for (key, value) in (paths).iteritems():
        tmp_dirs[key] = "{}/{}".format(tmpdir, paths[key])
        qa_dirs[key] = "{}/{}".format(qadir, paths[key])
    qc_stats = "{}/{}_stats.pkl".format(qadir, func_name)

    final_dirs = {}
    for (key, value) in finals.iteritems():
        final_dirs[key] = "{}/{}".format(outdir, finals[key])

    cmd = "mkdir -p {} {} {}".format(" ".join(tmp_dirs.values()),
                                     " ".join(qa_dirs.values()),
                                     " ".join(final_dirs.values()))
    mgu.execute_cmd(cmd)

    # Graphs are different because of multiple parcellations
    if isinstance(labels, list):
        label_name = [mgu.get_filename(x) for x in labels]
        for label in label_name:
            cmd = "mkdir -p {}/{} {}/{} {}/{}"
            cmd = cmd.format(final_dirs['ts_roi'], label, final_dirs['conn'],
                             label, qa_dirs['ts_roi'], label)
            mgu.execute_cmd(cmd)
    else:
        label_name = mgu.get_filename(labels)
        label = label_name
        cmd = "mkdir -p {}/{} {}/{} {}/{}"
        cmd = cmd.format(final_dirs['ts_roi'], label, final_dirs['conn'],
                         label, qa_dirs['ts_roi'], label)
        mgu.execute_cmd(cmd)

    # Create derivative output file names
    nuis_func = "{}/{}_nuis.nii.gz".format(tmp_dirs['nuis'], func_name)
    voxel_ts = "{}/timeseries/voxel/{}_voxel.npz".format(outdir, func_name)

    # Again, connectomes are different
    connectomes = ["{}/connectomes/{}/{}_{}.{}".format(outdir, x, func_name,
                                                       x, fmt)
                   for x in label_name]
    roi_ts = ["{}/{}/{}_{}.npz".format(final_dirs['ts_roi'], x, func_name, x)
              for x in label_name]
    print("ROI timeseries downsampled to given labels: " +
          ", ".join([x for x in roi_ts]))
    print("Connectomes downsampled to given labels: " +
          ", ".join([x for x in connectomes]))

    qc_func = qa_func()  # for quality control
    # ------- Nuisance Correction Steps ---------------------------- #
    print "Correcting Nuisance Variables..."
    nuis = mgn(func, t1w, nuis_func, tmp_dirs['nuis'],
               lv_mask, mc_params)
    nuis.nuis_correct(highpass=highpass, lowpass=lowpass, cc=cc,
                      csf_mean=csf_mean, wm_mean=wm_mean, mot=mot,
                      trend=trend)

    qc_func.nuisance_qa(nuis, qa_dirs['nuis'])


    # ------ ROI Timeseries Steps ---------------------------------- #
    for idx, label in enumerate(label_name):
        print "Extracting ROI timeseries for " + label + " parcellation..."
        ts = mgts().roi_timeseries(nuis_func, labels[idx], roi_ts[idx])
        labeldir = "{}/{}".format(qa_dirs['ts_roi'], label)
        connectome = mgg(ts.shape[0], labels[idx], sens="func")
        connectome.cor_graph(ts)
        connectome.summary()
        connectome.save_graph(connectomes[idx], fmt=fmt)
        qc_func.roi_ts_qa(roi_ts[idx], func, t1w,
                          labels[idx], labeldir)

    print("Execution took: {}".format(datetime.now() - startTime))
    if clean:
        cmd = "rm -rf {}".format(tmpdir)
        mgu.execute_cmd(cmd)
    print("Complete!")


def main():
    parser = ArgumentParser(description="This is an end-to-end connectome"
                            " estimation pipeline from sMRI and DTI images")
    parser.add_argument("func", action="store", help="Nifti fMRI 4d EPI.")
    parser.add_argument("t1w", action="store", help="Nifti aMRI T1w image.")
    parser.add_argument("lv_mask", action="store", help="Nifti binary mask of"
                        " lateral ventricles in atlas space.")
    parser.add_argument("outdir", action="store", help="Path to which"
                        " derivatives will be stored")
    parser.add_argument("labels", action="store", nargs="*", help="Nifti"
                        " labels of regions of interest in atlas space")
    parser.add_argument('--mc_params', action='store', help='motion parameters file.',
                        type=str, default=None)
    parser.add_argument("--highpass", action="store", help="the cutoff for highpass"
                        " filtering. None if not desired", default=0.01, type=float)
    parser.add_argument("--lowpass", action="store", help="the cutoff for lowpass"
                        " filtering. None if not desired.", default=None, type=float)
    parser.add_argument("--cc", action="store", help="the number of components for"
                        " aCompCor. None if not desired.", default=5, type=int)
    parser.add_argument("--csf_mean", action="store_true", help="whether to use csf mean"
                        " signal as a regressor.", default=False)
    parser.add_argument("--wm_mean", action="store_true", help="whether to use wm mean"
                        " signal as a regressor.", default=False)
    parser.add_argument("--mot", action="store", help="the number of regressors"
                        " to use from motion statistics. Uses the 6 x, y, z trans/rot"
                        " regressors if 6, uses friston 24 param model if 24, and does"
                        " not use motion regressors if none.", type=int, default=24,
                        choices=[None, 6, 24])
    parser.add_argument("--trend", action="store", help="the type of trendline to"
                        " use to remove scanner drifting.", default='quad',
                        choices=[None, 'lin', 'quad'])
    parser.add_argument("-c", "--clean", action="store_true", default=False,
                        help="Whether or not to delete intemediates")
    parser.add_argument("-f", "--fmt", action="store", default='gpickle',
                        help="Determines connectome output format")
    result = parser.parse_args()

    # Create output directory
    print "Creating output directory: {}".format(result.outdir)
    print "Creating output temp directory: {}/tmp".format(result.outdir)
    mgu.execute_cmd("mkdir -p {} {}/tmp".format(result.outdir, result.outdir))

    multinuis_pipeline(result.func, result.t1w, result.lv_mask, result.labels, result.outdir, 
                       clean=result.clean, fmt=result.fmt, mc_params=result.mc_params,
                       highpass=result.highpass, lowpass=result.lowpass,
                       cc=result.cc, csf_mean=result.csf_mean, wm_mean=result.wm_mean,
                       mot=result.mot, trend=result.trend)

if __name__ == "__main__":
    main()
