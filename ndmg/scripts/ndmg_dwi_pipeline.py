#!/usr/bin/env/python
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

# ndmg_dwi_pipeline.py
# Created by Greg Kiar and Will Gray Roncal on 2016-01-27.
# Email: gkiar@jhu.edu, wgr@jhu.edu
# Edited by Eric Bridgeford on 2017-07-13.

from __future__ import print_function

from argparse import ArgumentParser
from datetime import datetime
from subprocess import Popen, PIPE
from ndmg.stats.qa_regdti import *
from ndmg.stats.qa_tensor import *
from ndmg.stats.qa_fibers import *
import ndmg.utils as mgu
import ndmg.register as mgr
import ndmg.track as mgt
import ndmg.graph as mgg
import ndmg.preproc as mgp
import numpy as np
import nibabel as nb
import os
from ndmg.graph import biggraph as ndbg
import traceback

os.environ["MPLCONFIGDIR"] = "/tmp/"


def ndmg_dwi_worker(dwi, bvals, bvecs, mprage, atlas, mask, labels, outdir,
                    clean=False, fmt='gpickle', bg=False):
    """
    Creates a brain graph from MRI data
    """
    startTime = datetime.now()

    # Create derivative output directories
    dwi_name = mgu.get_filename(dwi)
    
    paths = {'preproc': 'dwi/preproc',
             'treg_d': 'dwi/reg/template',
             'sreg_d': 'dwi/reg/self',
             'sreg_a': 'dwi/reg/self',
             'treg_a': 't1w/reg/template',
             'tensors': 'dwi/tensors',
             'fibers': 'fibers'}
    finals = {'conn': 'connectomes'}

    tmpdir = '{}/tmp'.format(outdir)
    qadir = "{}/qa".format(outdir)

    tmp_dirs = {}
    qa_dirs = {}
    for (key, value) in (paths).iteritems():
        tmp_dirs[key] = "{}/{}".format(tmpdir, paths[key])
        qa_dirs[key] = "{}/{}".format(qadir, paths[key])

    final_dirs = {}

    # iterate over the items that we might want to clean out
    # if we need to clean them, just put the tmp directory into
    # their output since that directory is automatically deleted
    # otherwise, put the file in the standard output spec
    # and it will be preserved
    if not clean:
        finalpath = outdir
    else:
        finalpath = tmpdir

    for (key, value) in paths.iteritems():
        final_dirs[key] = "{}/{}".format(finalpath, paths[key])
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
            cmd = "mkdir -p {}/{}".format(final_dirs['conn'], label)
            mgu.execute_cmd(cmd)
    else:
        label_name = mgu.get_filename(labels)
        label = label_name
        cmd = "mkdir -p {}/{}".format(final_dirs['conn'], label)
        mgu.execute_cmd(cmd)

    # Create derivative output file names
    aligned_dwi = "{}/{}_{}".format(final_dirs['treg_d'], dwi_name,
                                    "_aligned.nii.gz")
    tensors = "{}/{}_{}".format(final_dirs['tensors'], dwi_name,
                                "_tensors.npz")
    fibers = "{}/{}_{}".format(final_dirs['fibers'], dwi_name, "_fibers.npz")
    print("This pipeline will produce the following derivatives...")
    print("DTI volume registered to atlas: " + aligned_dwi)
    print("Diffusion tensors in atlas space: " + tensors)
    print("Fiber streamlines in atlas space: " + fibers)

    # Again, graphs are different
    graphs = ["{}/{}/{}_{}.{}".format(final_dirs['conn'], x, dwi_name, x, fmt)
              for x in label_name]
    print("Graphs of streamlines downsampled to given labels: " +
          ", ".join([x for x in graphs]))

    # Creates gradient table from bvalues and bvectors
    print("Generating gradient table...")
    dwi1 = "{}/{}_{}".format(tmp_dirs['preproc'], dwi_name, "t1.nii.gz")
    bvecs1 = "{}/{}_{}".format(tmp_dirs['preproc'], dwi_name, "1.bvec")
    mgp.rescale_bvec(bvecs, bvecs1)
    gtab = mgu.load_bval_bvec_dwi(bvals, bvecs1, dwi, dwi1)

    # Align DTI volumes to Atlas
    print("Aligning volumes...")
    mgr().dti2atlas(dwi1, gtab, mprage, atlas, aligned_dwi, tmp_dirs)
    b0loc = np.where(gtab.b0s_mask)[0][0]
    reg_dti_pngs(aligned_dwi, b0loc, atlas, outdir+"/qa/reg_dwi/")

    print("Beginning tractography...")
    # Compute tensors and track fiber streamlines
    tens, tracks = mgt().eudx_basic(aligned_dwi, mask, gtab, stop_val=0.2)
    tensor2fa(tens, tensors, aligned_dwi, outdir+"/tensors/",
              outdir+"/qa/tensors/")

    # As we've only tested VTK plotting on MNI152 aligned data...
    if nb.load(mask).get_data().shape == (182, 218, 182):
        try:
            visualize_fibs(tracks, fibers, mask, outdir+"/qa/fibers/", 0.02)
        except:
            print("Fiber QA failed - VTK for Python not configured properly.")

    # And save them to disk
    np.savez(tensors, tens)
    np.savez(fibers, tracks)

    # Generate big graphs from streamlines
    if bg:
        print("Making Big Graph...")
        fibergraph = "{}/biggraph/{}_bg.edgelist".format(outdir, dwi_name)
        cmd = "mkdir -p {}/biggraph".format(outdir)
        mgu.execute_cmd(cmd)
        bg1 = ndbg()
        bg1.make_graph(tracks)
        bg1.save_graph(fibergraph)

    # Generate graphs from streamlines for each parcellation
    for idx, label in enumerate(label_name):
        print("Generating graph for " + label + " parcellation...")

        labels_im = nb.load(labels[idx])
        g1 = mgg(len(np.unique(labels_im.get_data()))-1, labels[idx])
        g1.make_graph(tracks)
        g1.summary()
        g1.save_graph(graphs[idx], fmt=fmt)

    print("Execution took: " + str(datetime.now() - startTime))

    # Clean temp files
    if clean:
        print("Cleaning up intermediate files... ")
        cmd = "rm -rf {}".format(tmpdir)
        mgu.execute_cmd(cmd)

    print("Complete!")
    pass


def ndmg_dwi_pipeline(dwi, bvals, bvecs, mprage, atlas, mask, labels, outdir,
                      clean=False, fmt='gpickle', bg=False):
    """
    A wrapper for the worker to make our pipeline more robust to errors.
    """
    try:
        ndmg_dwi_worker(dwi, bvals, bvecs, mprage, atlas, mask, labels, outdir,
                        clean, fmt, bg)
    except Exception, e:
        print(traceback.format_exc())
    return


def main():
    parser = ArgumentParser(description="This is an end-to-end connectome \
                            estimation pipeline from sMRI and DTI images")
    parser.add_argument("dwi", action="store", help="Nifti DTI image stack")
    parser.add_argument("bval", action="store", help="DTI scanner b-values")
    parser.add_argument("bvec", action="store", help="DTI scanner b-vectors")
    parser.add_argument("mprage", action="store", help="Nifti T1 MRI image")
    parser.add_argument("atlas", action="store", help="Nifti T1 MRI atlas")
    parser.add_argument("mask", action="store", help="Nifti binary mask of \
                        brain space in the atlas")
    parser.add_argument("outdir", action="store", help="Path to which \
                        derivatives will be stored")
    parser.add_argument("labels", action="store", nargs="*", help="Nifti \
                        labels of regions of interest in atlas space")
    parser.add_argument("-c", "--clean", action="store_true", default=False,
                        help="Whether or not to delete intemediates")
    parser.add_argument("-f", "--fmt", action="store", default='gpickle',
                        help="Determines graph output format")
    parser.add_argument("-b", "--bg", action="store_true", default=False,
                        help="whether or not to produce voxelwise big graph")
    result = parser.parse_args()

    # Create output directory
    cmd = "mkdir -p " + result.outdir + " " + result.outdir + "/tmp"
    print("Creating output directory: " + result.outdir)
    print("Creating output temp directory: " + result.outdir + "/tmp")
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    p.communicate()

    ndmg_dwi_pipeline(result.dwi, result.bval, result.bvec, result.mprage,
                      result.atlas, result.mask, result.labels, result.outdir,
                      result.clean, result.fmt, result.bg)


if __name__ == "__main__":
    main()
