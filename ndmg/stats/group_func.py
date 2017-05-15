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

# group_func.py
# Created by Eric W Bridgeford on 2016-04-03.
# Email: ebridge2@jhu.edu

import pickle
from ndmg.stats.qa_func import qa_func as fqa
import numpy as np
import ndmg.utils as mgu
import os
import plotly.offline as pyo
from plotly.tools import FigureFactory as ff
from ndmg.utils import loadGraphs
from ndmg.stats.qa_graphs import compute_metrics
from ndmg.stats.qa_graphs_plotting import make_panel_plot
import networkx as nx
from ndmg.stats.plotly_helper import *


class group_func(object):
    def __init__(self, basedir, outdir, atlas=None, dataset=None):
        """
        A class for group level quality control.

        **Positional Arguments:**
            - basedir:
                - the ndmg-formatted functional outputs.
                  should have a qa/ folder contained within it.
            - outdir:
                - the directory to place all group level quality control.
            - dataset:
                - an optional parameter for the name of the dataset
                  to be present in the quality control output filenames.
        """
        print atlas
        self.ndmgdir = basedir
        self.qadir = "{}/qa".format(self.ndmgdir)
        self.outdir = outdir
        self.conn_dir = "{}/connectomes".format(self.ndmgdir)
        self.dataset = dataset
        self.atlas = atlas
        (self.qa_files, self.subs) = self.get_qa_files()
        self.connectomes = self.get_connectomes()
        self.qa_objects = self.load_qa()
        self.group_level_analysis()
        self.connectome_analysis()
        pass

    def get_qa_files(self):
        """
        A function to load the relevant quality assessment files,
        for all the subjects we have in our study, given a properly-formatted
        ndmg functional directory.
        """
        qa_files = []
        subs = []
        for sub in os.listdir(self.qadir):
            sub_qa = "{}/{}/{}_stats.pkl".format(self.qadir, sub, sub)
            # if the files exists, add it to our qa_files
            if os.path.isfile(sub_qa):
                qa_files.append(sub_qa)
                subs.append(sub)
        return (qa_files, subs)

    def get_connectomes(self):
        """
        A function to load the relevant connectomes for all of the subjects
        for each parcellation we have.
        """
        connectomes = {}
        for label in os.listdir(self.conn_dir):
            print label
            this_label = []
            label_dir = "{}/{}".format(self.conn_dir, label)
            for connectome in os.listdir(label_dir):
                conn_path = "{}/{}".format(label_dir, connectome)
                if os.path.isfile(conn_path):
                    this_label.append(conn_path)
            connectomes[label] = this_label
        return connectomes

    def load_qa(self):
        """
        A function to load the quality control objects.
        """
        qa_objects = []
        for qa_file in self.qa_files:
            # load the qa objects as qa_func objects
            qa_objects.append(fqa.load(qa_file))
        return qa_objects

    def group_level_analysis(self):
        """
        A function to perform group level analysis after loading the
        functional qa objects properly.
        """
        self.group_reg()
        self.group_motion()

    def group_reg(self):
        """
        A function that performs group level registration quality control.
        """
        regdir = "{}/{}".format(self.outdir, "reg")
        cmd = "mkdir -p {}".format(regdir)
        mgu.execute_cmd(cmd)

        self_reg_sc = []
        temp_reg_sc = []
        cnr = []
        snr = []
        for sub in self.qa_objects:
            self_reg_sc.append(sub.self_reg_sc)
            temp_reg_sc.append(sub.temp_reg_sc)
            cnr.append(sub.cnr)
            snr.append(sub.snr)

        fig_cnr = plot_rugdensity(cnr)
        fig_snr = plot_rugdensity(snr)
        fig_sreg = plot_rugdensity(self_reg_sc)
        fig_treg = plot_rugdensity(temp_reg_sc)

        figs = [fig_cnr, fig_snr, fig_sreg, fig_treg]
        names = ['temporal Contrast to Noise Ratio', 'temporal Signal to Noise Ratio',
                 'Self Registration Score', 'Template Registration Score']
        ylab = ['Density', 'Density', 'Density', 'Density']
        xlab = ['Ratio', 'Ratio', 'Score', 'Score']
        traces = [fig_to_trace(fig) for fig in figs]
 
        fname_multi = "registration_qa.html"
        # if a dataset name is provided, add it to the name
        if self.dataset is not None:
            fname_multi = "{}_{}".format(self.dataset, fname_multi)
        fname_multi = "{}/{}".format(regdir, fname_multi)
        multi = traces_to_panels(traces, names=names, ylabs=ylab, xlabs=xlab)
        pyo.plot(multi, validate=False, filename=fname_multi)
        pass

    def group_motion(self):
        """
        A function that performs group level motion corrective quality control.
        """
        mcdir = "{}/{}".format(self.outdir, "mc")
        cmd = "mkdir -p {}".format(mcdir)
        mgu.execute_cmd(cmd)

        trans_abs = np.zeros((len(self.qa_objects)))
        trans_rel = np.zeros((len(self.qa_objects)))
        trans_abs_gt = np.zeros((len(self.qa_objects)))
        trans_rel_gt = np.zeros((len(self.qa_objects)))

        for i, sub in enumerate(self.qa_objects):
            abs_m = np.linalg.norm(sub.abs_pos[:, 3:6], axis=1)
            rel_m = np.linalg.norm(sub.rel_pos[:, 3:6], axis=1)
            trans_abs[i] = np.mean(abs_m)
            trans_rel[i] = np.mean(rel_m)
            trans_abs_gt[i] = np.sum(abs_m > 0.2)
            trans_rel_gt[i] = np.sum(rel_m > 0.1)

        fig_abs = plot_rugdensity(trans_abs)
        fig_rel = plot_rugdensity(trans_rel)
        fig_abs_gt = plot_rugdensity(trans_abs_gt)
        fig_rel_gt = plot_rugdensity(trans_rel_gt)

        figs = [fig_abs, fig_rel, fig_abs_gt, fig_rel_gt]
        names = ['Average Absolute Translational Motion', 'Average Relative Translational Motion',
                 'Number of Absolute Motions > 0.2 mm', 'Number of Relative Motions > 0.1 mm']
        ylab = ['Density', 'Density', 'Density', 'Density']
        xlab = ['Average Motion (mm)', 'Average Motion (mm)',
                'Number of Volumes', 'Number of Volumes']
        traces = [fig_to_trace(fig) for fig in figs]

        fname_multi = "motion_correction.html"

        # if a dataset name is provided, add it to the name
        if self.dataset is not None:
            fname_multi = "{}_{}".format(self.dataset,
                                         fname_multi)
        fname_multi = "{}/{}".format(mcdir, fname_multi)

        multi = traces_to_panels(traces, names=names, ylabs=ylab, xlabs=xlab)
        pyo.plot(multi, validate=False, filename=fname_multi) 
        pass

    def connectome_analysis(self, thr=0.85, minimal=False, log=False,
                            hemispheres=False):
        """
        A function to threshold and binarize the connectomes.
        Presently just thresholds to reference correlation of
        setting all edges below 0.3 to 0, and those greater to 1.
        This value of 0.3 was generally the highest performing in
        discriminability analyses.

        **Positional Arguments:**
            - thr:
                - the threshold to binarize below.
        """
        self.graph_dir = "{}/connectomes".format(self.outdir)
        cmd = "mkdir -p {}".format(self.graph_dir)
        mgu.execute_cmd(cmd)
        for label, raw_conn_files in self.connectomes.iteritems():
            print("Parcellation: {}".format(label))
            label_raw = loadGraphs(raw_conn_files)
            label_connectomes = {}
            label_dir = "{}/{}".format(self.graph_dir, label)
            tmp_dir = "{}/connectomes".format(label_dir)
            # verify that directories exist
            cmd = "mkdir -p {}".format(label_dir)
            mgu.execute_cmd(cmd)
            cmd = "mkdir -p {}".format(tmp_dir)
            mgu.execute_cmd(cmd)
            for subj, raw in label_raw.iteritems():
                # loop over edges to threshold
                raw_mtx = nx.to_numpy_matrix(raw)
                cor_thr = np.percentile(raw_mtx, thr*100) 
                for u, v, d in raw.edges(data=True):
                    # threshold connectomes by removing weights
                    # above the threshold
                    if d['weight'] < cor_thr:
                        raw.remove_edge(u, v)
                    else:
                         d['weight'] = 1
                # resave the thresholded connectomes
                gname = "{}/{}".format(tmp_dir, subj)
                nx.write_gpickle(raw, gname)
                # so our connectomes are in the format expected by
                # graphing qa
                label_connectomes[subj] = gname
            compute_metrics(label_connectomes.values(), label_dir, label)
            outf = os.path.join(label_dir, "{}_plot".format(label))
            make_panel_plot(label_dir, outf, dataset=self.dataset,
                            atlas=label, minimal=minimal,
                            log=log, hemispheres=hemispheres)
            print outf
        pass

