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

# qa_func.py
# Created by Eric W Bridgeford on 2016-06-08.
# Email: ebridge2@jhu.edu

import nibabel as nb
import sys
import re
import os.path
import matplotlib
import numpy as np
from ndmg.utils import utils as mgu
from ndmg.stats.func_qa_utils import plot_timeseries, plot_signals, \
    registration_score
from ndmg.stats.qa_reg import reg_mri_pngs, plot_brain, plot_overlays
from ndmg.register.register import func_register as ndfr
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly as py
import plotly.offline as offline
import pickle


class qa_func(object):
    def __init__(self):
        pass

    @staticmethod
    def load(filename):
        """
        A function for loading a qa_func object, so that we
        can perform group level quality control easily.

        **Positional Arguments:**

            filename: the name of the pickle file containing
                our qa_func object
        """
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        f.close()
        return obj

    def save(self, filename):
        """
        A function for saving a qa_func object.

        **Positional Arguments:**

            filename: the name of the file we want to save to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        f.close()
        pass

    def preproc_qa(self, prep, qcdir=None):
        """
        A function for performing quality control given motion
        correction information. Produces plots of the motion correction
        parameters used.

        **Positional Arguments**

            prep:
                - the module used for preprocessing.
            scan_id:
                - the id of the subject.
            qcdir:
                - the quality control directory.
        """
        print "Performing QA for Preprocessing..."
        cmd = "mkdir -p {}".format(qcdir)
        mgu.execute_cmd(cmd)
        scanid = mgu.get_filename(prep.motion_func)

        mc_im = nb.load(prep.motion_func)
        mc_dat = mc_im.get_data()

        mcfig = plot_brain(mc_dat.mean(axis=3), minthr=10)
        nvols = mc_dat.shape[3]

        fnames = {}
        fnames['trans'] = "{}_trans.html".format(scanid)
        fnames['rot'] = "{}_rot.html".format(scanid)

        par_file = prep.mc_params
        mc_file = "{}/{}_stats.txt".format(qcdir, scanid)

        abs_pos = np.zeros((nvols, 6))
        rel_pos = np.zeros((nvols, 6))
        with open(par_file) as f:
            counter = 0
            for line in f:
                abs_pos[counter, :] = [float(i) for i in re.split("\\s+",
                                                                  line)[0:6]]
                if counter > 0:
                    rel_pos[counter, :] = np.subtract(abs_pos[counter, :],
                                                      abs_pos[counter-1, :])
                counter += 1

        trans_abs = np.linalg.norm(abs_pos[:, 3:6], axis=1)
        trans_rel = np.linalg.norm(rel_pos[:, 3:6], axis=1)
        rot_abs = np.linalg.norm(abs_pos[:, 0:3], axis=1)
        rot_rel = np.linalg.norm(rel_pos[:, 0:3], axis=1)

        self.abs_pos = abs_pos
        self.rel_pos = rel_pos

        fmc_list = []
        fmc_list.append(py.graph_objs.Scatter(x=range(0, nvols), y=trans_abs,
                                              mode='lines', name='absolute'))
        fmc_list.append(py.graph_objs.Scatter(x=range(0, nvols), y=trans_rel,
                                              mode='lines', name='relative'))
        layout = dict(title='Estimated Displacement',
                      xaxis=dict(title='Timepoint', range=[0, nvols]),
                      yaxis=dict(title='Movement (mm)'))
        fmc = dict(data=fmc_list, layout=layout)

        disp_path = "{}/{}_disp.html".format(qcdir, scanid)
        offline.plot(fmc, filename=disp_path, auto_open=False)
        ftrans_list = []
        ftrans_list.append(py.graph_objs.Scatter(x=range(0, nvols),
                                                 y=abs_pos[:, 3],
                                                 mode='lines', name='x'))
        ftrans_list.append(py.graph_objs.Scatter(x=range(0, nvols),
                                                 y=abs_pos[:, 4],
                                                 mode='lines', name='y'))
        ftrans_list.append(py.graph_objs.Scatter(x=range(0, nvols),
                                                 y=abs_pos[:, 5],
                                                 mode='lines', name='z'))
        layout = dict(title='Translational Motion Parameters',
                      xaxis=dict(title='Timepoint', range=[0, nvols]),
                      yaxis=dict(title='Translation (mm)'))
        ftrans = dict(data=ftrans_list, layout=layout)
        trans_path = "{}/{}_trans.html".format(qcdir, scanid)
        offline.plot(ftrans, filename=trans_path, auto_open=False)

        frot_list = []
        frot_list.append(py.graph_objs.Scatter(x=range(0, nvols),
                                               y=abs_pos[:, 0],
                                               mode='lines', name='x'))
        frot_list.append(py.graph_objs.Scatter(x=range(0, nvols),
                                               y=abs_pos[:, 1],
                                               mode='lines', name='y'))
        frot_list.append(py.graph_objs.Scatter(x=range(0, nvols),
                                               y=abs_pos[:, 2],
                                               mode='lines', name='z'))
        layout = dict(title='Rotational Motion Parameters',
                      xaxis=dict(title='Timepoint', range=[0, nvols]),
                      yaxis=dict(title='Rotation (rad)'))
        frot = dict(data=frot_list, layout=layout)
        rot_path = "{}/{}_rot.html".format(qcdir, scanid)
        offline.plot(frot, filename=rot_path, auto_open=False)

        # Motion Statistics
        mean_abs = np.mean(abs_pos, axis=0)  # column wise means per param
        std_abs = np.std(abs_pos, axis=0)
        max_abs = np.max(np.abs(abs_pos), axis=0)
        mean_rel = np.mean(rel_pos, axis=0)
        std_rel = np.std(rel_pos, axis=0)
        max_rel = np.max(np.abs(rel_pos), axis=0)

        fstat = open(mc_file, 'w')
        fstat.write("Motion Statistics\n")

        absrel = ["absolute", "relative"]
        transrot = ["motion", "rotation"]
        list1 = [max(trans_abs), np.mean(trans_abs), np.sum(trans_abs > 1),
                 np.sum(trans_abs > 5), mean_abs[3], std_abs[3], max_abs[3],
                 mean_abs[4], std_abs[4], max_abs[4], mean_abs[5],
                 std_abs[5], max_abs[5]]
        list2 = [max(trans_rel), np.mean(trans_rel), np.sum(trans_rel > 1),
                 np.sum(trans_rel > 5), mean_abs[3], std_rel[3], max_rel[3],
                 mean_abs[4], std_rel[4], max_rel[4], mean_abs[5],
                 std_rel[5], max_rel[5]]
        list3 = [max(rot_abs), np.mean(rot_abs), 0, 0, mean_abs[0],
                 std_abs[0], max_abs[0], mean_abs[1], std_abs[1],
                 max_abs[1], mean_abs[2], std_abs[2], max_abs[2]]
        list4 = [max(rot_rel), np.mean(rot_rel), 0, 0, mean_rel[0],
                 std_rel[0], max_rel[0], mean_rel[1], std_rel[1],
                 max_rel[1], mean_rel[2], std_rel[2], max_rel[2]]
        lists = [list1, list2, list3, list4]
        headinglist = ["Absolute Translational Statistics>>\n",
                       "Relative Translational Statistics>>\n",
                       "Absolute Rotational Statistics>>\n",
                       "Relative Rotational Statistics>>\n"]
        x = 0

        for motiontype in transrot:
            for measurement in absrel:
                fstat.write(headinglist[x])
                fstat.write("Max " + measurement + " " + motiontype +
                            ": %.4f\n" % lists[x][0])
                fstat.write("Mean " + measurement + " " + motiontype +
                            ": %.4f\n" % lists[x][1])
                if motiontype == "motion":
                    fstat.write("Number of " + measurement + " " + motiontype +
                                "s > 1mm: %.4f\n" % lists[x][2])
                    fstat.write("Number of " + measurement + " " + motiontype +
                                "s > 5mm: %.4f\n" % lists[x][3])
                fstat.write("Mean " + measurement + " x " + motiontype +
                            ": %.4f\n" % lists[x][4])
                fstat.write("Std " + measurement + " x " + motiontype +
                            ": %.4f\n" % lists[x][5])
                fstat.write("Max " + measurement + " x " + motiontype +
                            ": %.4f\n" % lists[x][6])
                fstat.write("Mean " + measurement + " y " + motiontype +
                            ": %.4f\n" % lists[x][7])
                fstat.write("Std " + measurement + " y " + motiontype +
                            ": %.4f\n" % lists[x][8])
                fstat.write("Max " + measurement + " y " + motiontype +
                            ": %.4f\n" % lists[x][9])
                fstat.write("Mean " + measurement + " z " + motiontype +
                            ": %.4f\n" % lists[x][10])
                fstat.write("Std " + measurement + " z " + motiontype +
                            ": %.4f\n" % lists[x][11])
                fstat.write("Max " + measurement + " z " + motiontype +
                            ": %.4f\n" % lists[x][12])
                x = x + 1

        fstat.close()
        return

    def self_reg_qa(self, freg, qa_dirs):
        """
        A function that produces self-registration quality control figures.

        **Positional Arguments:**

            freg:
                - the func_register object from registration.
            sreg_func_dir:
                - the directory to place functional qc images.
        """
        print "Performing QA for Self-Registration..."
        (sreg_sc, sreg_fig) = registration_score(
            freg.sreg_brain,
            freg.t1w_brain
        )
        self.self_reg_sc = sreg_sc
        sreg_f_final = "{}/{}_score_{:.0f}".format(
            qa_dirs['sreg_f'],
            freg.sreg_strat,
            self.self_reg_sc*1000
        )
        sreg_a_final = "{}/{}_score_{:.0f}".format(
            qa_dirs['sreg_a'],
            freg.sreg_strat,
            self.self_reg_sc*1000
        )
        cmd = "mkdir -p {} {}".format(sreg_f_final, sreg_a_final)
        mgu.execute_cmd(cmd)
        func_name = mgu.get_filename(freg.sreg_brain)
        t1w_name = mgu.get_filename(freg.t1w)
        sreg_fig.savefig(
            "{}/{}_epi2t1w.png".format(sreg_f_final, func_name)  
        )
        # provide qc for the skull stripping step
        t1brain_dat = nb.load(freg.t1w_brain).get_data()
        t1_dat = nb.load(freg.t1w).get_data()
        freg_qual = plot_overlays(t1_dat, t1brain_dat)
        fname = "{}/{}_bet_quality.png".format(sreg_a_final, t1w_name)
        freg_qual.savefig(fname)
        plt.close()
        pass

    def temp_reg_qa(self, freg, qa_dirs):
        """
        A function that produces self-registration quality control figures.

        **Positional Arguments:**

            freg:
                - the functional registration object.
            qa_dirs:
                - a dictionary of the directories to place qa files.
        """
        print "Performing QA for Template-Registration..."
        (treg_sc, treg_fig) = registration_score(
            freg.taligned_epi,
            freg.atlas_brain
        )
        self.temp_reg_sc = treg_sc
        treg_f_final = "{}/{}_score_{:.0f}".format(
            qa_dirs['treg_f'],
            freg.treg_strat,
            self.temp_reg_sc*1000
        )
        treg_a_final = "{}/{}_score_{:.0f}".format(
            qa_dirs['treg_a'],
            freg.treg_strat,
            self.temp_reg_sc*1000
        )
        cmd = "mkdir -p {} {}".format(treg_f_final, treg_a_final)
        mgu.execute_cmd(cmd)
        func_name = mgu.get_filename(freg.taligned_epi)
        treg_fig.savefig(
            "{}/{}_epi2temp.png".format(treg_f_final, func_name)  
        )
        t1w_name = mgu.get_filename(freg.taligned_t1w)
        t1w2temp_fig = plot_overlays(freg.taligned_t1w, freg.atlas_brain)
        t1w2temp_fig.savefig(
            "{}/{}_t1w2temp.png".format(treg_a_final, t1w_name)
        )
        self.voxel_qa(freg.taligned_epi, freg.atlas_mask, treg_f_final)

    def voxel_qa(self, func, mask, qadir):
        """
        A function to compute voxelwise statistics, such as voxelwise mean,
        voxelwise snr, voxelwise cnr, for an image, and produce related
        qa plots.

        **Positional Arguments:**

            func:
                - the path to the functional image we want statistics for.
            mask:
                - the path to the anatomical mask.
            qadir:
                - the directory to place qa images.
        """
        # estimating mean signal intensity and deviation in brain/non-brain
        fmri = nb.load(func)
        mask = nb.load(mask)
        fmri_dat = fmri.get_data()
        mask_dat = mask.get_data()

        # threshold to identify the brain and non-brain regions
        brain = fmri_dat[mask_dat > 0, :]
        non_brain = fmri_dat[mask_dat == 0, :]
        # identify key statistics
        mean_brain = brain.mean()
        std_nonbrain = np.nanstd(non_brain)
        std_brain = np.nanstd(brain)
        self.snr = mean_brain/std_nonbrain
        self.cnr = std_brain/std_nonbrain

        scanid = mgu.get_filename(func)

        np.seterr(divide='ignore', invalid='ignore')
        mean_ts = fmri_dat.mean(axis=3)
        snr_ts = np.divide(mean_ts, std_nonbrain)
        cnr_ts = np.divide(np.nanstd(fmri_dat, axis=3), std_nonbrain)

        plots = {}
        plots["mean"] = plot_brain(mean_ts, minthr=10)
        plots["snr"] = plot_brain(snr_ts, minthr=10)
        plots["cnr"] = plot_brain(cnr_ts, minthr=10)
        for plotname, plot in plots.iteritems():
            fname = "{}/{}_{}.png".format(qadir, scanid, plotname)
            plot.savefig(fname, format='png')
            plt.close()
        pass

    def nuisance_qa(self, nuisobj, qcdir):
        """
        A function to assess the quality of nuisance correction.

        **Positional Arguments**

            nuisobj:
                - the nuisance correction object.
            qcdir:
                - the directory to place quality control images.
        """
        print "Performing QA for Nuisance..."
        maskdir = "{}/{}".format(qcdir, "masks")
        glmdir = "{}/{}".format(qcdir, "glm_correction")
        fftdir = "{}/{}".format(qcdir, "filtering")

        cmd = "mkdir -p {} {} {}".format(qcdir, maskdir, glmdir)
        mgu.execute_cmd(cmd)

        anat_name = mgu.get_filename(nuisobj.smri)
        t1w_dat = nb.load(nuisobj.smri).get_data()
        masks = [nuisobj.lv_mask, nuisobj.wm_mask, nuisobj.gm_mask,
                 nuisobj.er_wm_mask]
        masknames = ["csf_mask", "wm_mask", "gm_mask", "eroded_wm_mask"]
        # iterate over masks for existence and plot overlay if they exist
        # since that means they were used at some point
        for mask, maskname in zip(masks, masknames):
            if mask is not None:
                mask_dat = nb.load(mask).get_data()
                f_mask = plot_overlays(t1w_dat, mask_dat, min_val=0, max_val=1)
                fname_mask = "{}/{}_{}.png".format(maskdir, anat_name,
                                                   maskname)
                f_mask.savefig(fname_mask, format='png')
                plt.close()

        # GLM regressors
        glm_regs = [nuisobj.csf_reg, nuisobj.wm_reg, nuisobj.friston_reg]
        glm_names = ["csf", "wm", "friston"]
        glm_titles = ["CSF Regressors", "White-Matter Regressors",
                      "Friston Motion Regressors"]
        label_include = [True, True, False]
        for (reg, name, title, lab) in zip(glm_regs, glm_names, glm_titles,
                label_include):
            if reg is not None:
                regs = []
                labels = []
                for i in range(0, reg.shape[1]):
                    regs.append(reg[:, i])
                    labels.append('{} reg {}'.format(name, i))
                fig = plot_signals(regs, labels, title=title,
                                   xlabel='Timepoint', ylabel='Intensity',
                                   lab_incl=lab)
                fname_reg = "{}/{}_{}_regressors.png".format(glmdir,
                                                             anat_name,
                                                             name)
                fig.savefig(fname_reg, format='png')
                plt.close()
        # before glm compared with the signal removed and
        # signal after correction
        fig_glm_sig = plot_signals(
                [nuisobj.cent_nuis, nuisobj.glm_sig, nuisobj.glm_nuis],
                ['Before', 'Regressed Sig', 'After'],
                title='Impact of GLM Regression on Average GM Signal',
                xlabel='Timepoint',
                ylabel='Intensity'
        )
        fname_glm_sig = '{}/{}_glm_signal_cmp.png'.format(glmdir, anat_name)
        fig_glm_sig.savefig(fname_glm_sig, format='png')
        plt.close()

        # Frequency Filtering
        # start by just plotting the average fft of gm voxels and compare with
        # average fft after frequency filtering
        if nuisobj.fft_reg is not None:
            cmd = "mkdir -p {}".format(fftdir)
            mgu.execute_cmd(cmd)

            fig_fft_pow = plot_signals(
                    [nuisobj.fft_bef, nuisobj.fft_reg],
                    ['Before', 'After'],
                    title='Average Gray Matter Power Spectrum',
                    xlabel='Frequency',
                    ylabel='Power',
                    xax=nuisobj.freq_ra)
            fname_fft_pow = '{}/{}_fft_power.png'.format(fftdir, anat_name)
            fig_fft_pow.savefig(fname_fft_pow, format='png')
            plt.close()
            # plot the signal vs the regressed signal vs signal after
            fig_fft_sig = plot_signals(
                    [nuisobj.glm_nuis, nuisobj.fft_sig, nuisobj.fft_nuis],
                    ['Before', 'Regressed Sig', 'After'],
                    title='Impact of Frequency Filtering on Average GM Signal',
                    xlabel='Timepoint',
                    ylabel='Intensity')
            fname_fft_sig = '{}/{}_fft_signal_cmp.png'.format(
                    fftdir,
                    anat_name)
            fig_fft_sig.savefig(fname_fft_sig, format='png')
            plt.close()
        pass

    def roi_ts_qa(self, timeseries, func, anat, label, qcdir):
        """
        A function to perform ROI timeseries quality control.

        **Positional Arguments**

            timeseries:
                - a path to the ROI timeseries.
            func:
                - the functional image that has timeseries
                extract from it.
            anat:
                - the anatomical image that is aligned.
            label:
                - the label in which voxel timeseries will be
                downsampled.
            qcdir:
                - the quality control directory to place outputs.
        """
        print "Performing QA for ROI Timeseries..."
        cmd = "mkdir -p {}".format(qcdir)
        mgu.execute_cmd(cmd)

        reg_mri_pngs(anat, label, qcdir, minthr=10, maxthr=95)
        plot_timeseries(timeseries, qcdir=qcdir)
        pass

    def voxel_ts_qa(self, timeseries, voxel_func, atlas_mask, qcdir):
        """
        A function to analyze the voxel timeseries extracted.

        **Positional Arguments**

            voxel_func:
                - the functional timeseries that
              has voxel timeseries extracted from it.
            atlas_mask:
                - the mask under which
              voxel timeseries was extracted.
            qcdir:
                - the directory to place qc in.
        """
        print "Performing QA for Voxel Timeseries..."
        cmd = "mkdir -p {}".format(qcdir)
        mgu.execute_cmd(cmd)
        reg_mri_pngs(voxel_func, atlas_mask, qcdir,
                     loc=0, minthr=10, maxthr=95)
        self.voxel_qa(voxel_func, atlas_mask, qcdir)
        pass
