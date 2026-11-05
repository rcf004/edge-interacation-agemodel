#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:57:20 2024

@author: robfren
"""

from pathlib import Path
from collections.abc import Iterable
import os
import numpy as np
from scipy import stats
import pickle

import matplotlib.pyplot as plt


import pandas as pd
import seaborn as sns


from nilearn import surface
from nilearn import plotting
from nilearn import datasets

import warnings
import mat_manip as mm
import plotting_funcs as pf


#%% class setup

class Modelterm:
    def __init__(self, model_name, term_name, sub_beta, sub_pval):
        self.model_name = model_name
        self.term_name = term_name
        self.sub_beta = np.array(sub_beta)
        self.sub_pval = np.array(sub_pval)
            
    def edge_sigtest(self, thresh_level=0.001, FDR=True):
        # performed on vector
        
        self.edge_t, self.edge_sig = stats.ttest_1samp(self.sub_beta, popmean=0, axis=0)
        self.edge_sig[np.isnan(self.edge_sig)] = 1
        self.edge_t[np.isnan(self.edge_t)] = 0
        
        if FDR:
            self.edge_sig = stats.false_discovery_control(self.edge_sig)
        
        self.edge_sig_bin = np.where(self.edge_sig < thresh_level, 1, 0)
        self.edge_thresh = np.multiply(self.edge_sig_bin, self.edge_t)
        
    def createmats(self, ntotal, reorg_idx, trim):
        if self.term_name == 'nodal':
            # reshape subject level first
            self.sub_beta_mat = np.reshape(self.sub_beta, (self.sub_beta.shape[0], ntotal, ntotal))
            self.sub_pval_mat = np.reshape(self.sub_pval, (self.sub_beta.shape[0], ntotal, ntotal))
            # now group level
            self.edge_t_mat = np.reshape(self.edge_t, (ntotal, ntotal))
            self.edge_sig_mat = np.reshape(self.edge_sig, (ntotal, ntotal))
            self.edge_sig_bin_mat = np.reshape(self.edge_sig_bin, (ntotal, ntotal))
            self.edge_thresh_mat = np.reshape(self.edge_thresh, (ntotal, ntotal))
           
            
        elif self.term_name == 'edge':
            # reshape subject level first
            self.sub_beta_mat = np.apply_along_axis(mm.mask_based_utri2mat, 1, self.sub_beta, *[ntotal])
            self.sub_pval_mat = np.apply_along_axis(mm.mask_based_utri2mat, 1, self.sub_pval, *[ntotal])
            # now group level
            self.edge_t_mat = mm.mask_based_utri2mat(self.edge_t, ntotal)
            self.edge_sig_mat = mm.mask_based_utri2mat(self.edge_sig, ntotal)
            self.edge_sig_bin_mat = mm.mask_based_utri2mat(self.edge_sig_bin, ntotal)
            self.edge_thresh_mat = mm.mask_based_utri2mat(self.edge_thresh, ntotal)
            
        else:
            raise ValueError('term argument must be `edge` or `nodal`')
            
        # reorg sub level
        self.sub_beta_mat_reorg = np.array([mm.reorg(self.sub_beta_mat[i,:,:], reorg_idx, trim) for i in range(self.sub_beta_mat.shape[0])])
        self.sub_pval_mat_reorg = np.array([mm.reorg(self.sub_pval_mat[i,:,:], reorg_idx, trim) for i in range(self.sub_pval_mat.shape[0])])
        
        # reorg group level
        self.edge_t_reorg = mm.reorg(self.edge_t_mat, reorg_idx, trim)
        self.edge_sig_reorg = mm.reorg(self.edge_sig_mat, reorg_idx, trim)
        self.edge_sig_bin_reorg = mm.reorg(self.edge_sig_bin_mat, reorg_idx, trim)
        self.edge_thresh_reorg = mm.reorg(self.edge_thresh_mat, reorg_idx, trim)
    
    def spin_test(self, iters, rerun, spin_ord, bounds, alpha=0.05):
        if type(rerun) != bool:
            raise AttributeError('`rerun` must be True or False')
        
        spin_dir = f'data/null_spin_models/{self.model_name}'    
        if not os.path.exists(spin_dir):
            os.makedirs(spin_dir)

        if self.term_name == 'nodal':
            axis='single'                
        elif self.term_name == 'edge':
            axis='double'
        else:
            axis = ''
                
            
        if rerun:

            self.edge_thresh_pos = np.where(self.edge_thresh_reorg > 0, self.edge_thresh_reorg, 0)
            self.edge_thresh_neg = np.where(self.edge_thresh_reorg < 0, self.edge_thresh_reorg, 0)
            self.pos_count, self.pos_null = mm.spin_test(self.edge_thresh_pos, spin_ord, bounds, axis, iters=iters)
            self.neg_count, self.neg_null = mm.spin_test(self.edge_thresh_neg, spin_ord, bounds, axis, iters=iters)

            self.count, self.null = mm.spin_test(self.edge_thresh_reorg, spin_ord, bounds, axis, iters=iters)
            
            np.save(f'{spin_dir}/pos_count_{self.model_name}_{self.term_name}_{iters}.npy', self.pos_count)
            np.save(f'{spin_dir}/neg_count_{self.model_name}_{self.term_name}_{iters}.npy', self.neg_count)
            np.save(f'{spin_dir}/pos_null_{self.model_name}_{self.term_name}_{iters}.npy', self.pos_null)
            np.save(f'{spin_dir}/neg_null_{self.model_name}_{self.term_name}_{iters}.npy', self.neg_null)
            
            np.save(f'{spin_dir}/count_{self.model_name}_{self.term_name}_{iters}.npy', self.count)
            np.save(f'{spin_dir}/null_{self.model_name}_{self.term_name}_{iters}.npy', self.null)

        else:
            self.edge_thresh_pos = np.where(self.edge_thresh_reorg > 0, self.edge_thresh_reorg, 0)
            self.edge_thresh_neg = np.where(self.edge_thresh_reorg < 0, self.edge_thresh_reorg, 0)
            
            self.pos_count = np.load(f'{spin_dir}/pos_count_{self.model_name}_{self.term_name}_{iters}.npy')
            self.neg_count = np.load(f'{spin_dir}/neg_count_{self.model_name}_{self.term_name}_{iters}.npy')
            self.pos_null = np.load(f'{spin_dir}/pos_null_{self.model_name}_{self.term_name}_{iters}.npy')
            self.neg_null = np.load(f'{spin_dir}/neg_null_{self.model_name}_{self.term_name}_{iters}.npy')
            
            self.count = np.load(f'{spin_dir}/count_{self.model_name}_{self.term_name}_{iters}.npy')
            self.null = np.load(f'{spin_dir}/null_{self.model_name}_{self.term_name}_{iters}.npy')        
        
        self.pos_block = mm.check_null(self.pos_count, self.pos_null, axis, alpha=alpha)
        self.neg_block = mm.check_null(self.neg_count, self.neg_null, axis, alpha=alpha)
        self.block = mm.check_null(self.count, self.null, axis, alpha=alpha)
        
        if self.term_name == 'nodal':
            _temp = np.empty((self.pos_block.shape[1], self.pos_block.shape[1]))
            for i in range(self.pos_block.shape[1]):
                _temp[:,i] = self.pos_block[0,:]
            
            self.pos_block = _temp
            
            _temp = np.empty((self.neg_block.shape[1], self.neg_block.shape[1]))
            for i in range(self.neg_block.shape[1]):
                _temp[:,i] = self.neg_block[0,:]
                
            self.neg_block = _temp

            _temp = np.empty((self.block.shape[1], self.block.shape[1]))
            for i in range(self.block.shape[1]):
                _temp[:,i] = self.block[0,:]
                
            self.block = _temp
 
    def plot_block(self, bands, names, rot_override=None, **kwargs):
        
        outd = f'figs/matrix_plots/{self.model_name}'    
        if not os.path.exists(outd):
            os.makedirs(outd)
        
        # create pos/neg block masks from spin tests
        # TODO- wrap this using mm.expand block (should be as simple as a check if block is 1, make expanded 1)
        self._pos_blockmask = np.zeros(np.shape(self.edge_thresh_reorg))
        for row_start, row_end, blockr in zip(bands[:-1], bands[1:], range(self.pos_block.shape[0])):
            for col_start, col_end, blockc in zip(bands[:-1], bands[1:], range(self.pos_block.shape[1])):
                if self.pos_block[blockr, blockc] == 1:
                    self._pos_blockmask[row_start:row_end, col_start:col_end] = 1
        # negative
        self._neg_blockmask = np.zeros(np.shape(self.edge_thresh_reorg))
        for row_start, row_end, blockr in zip(bands[:-1], bands[1:], range(self.neg_block.shape[0])):
            for col_start, col_end, blockc in zip(bands[:-1], bands[1:], range(self.neg_block.shape[1])):
                if self.neg_block[blockr, blockc] == 1:
                    self._neg_blockmask[row_start:row_end, col_start:col_end] = 1
        
        self._blockmask = np.zeros(np.shape(self.edge_thresh_reorg))
        for row_start, row_end, blockr in zip(bands[:-1], bands[1:], range(self.block.shape[0])):
            for col_start, col_end, blockc in zip(bands[:-1], bands[1:], range(self.block.shape[1])):
                if self.block[blockr, blockc] == 1:
                    self._blockmask[row_start:row_end, col_start:col_end] = 1
               
        self._pos_mean = mm.reduce_block(self.edge_thresh_pos, bands, mm.nonzeromean)()
        self._neg_mean = mm.reduce_block(self.edge_thresh_neg, bands, mm.nonzeromean)()
        self._mean = mm.reduce_block(self.edge_thresh_reorg, bands, mm.nonzeromean)()

        #TODO- wrap this using expand block
        self._pos_blockmean = np.zeros(np.shape(self.edge_thresh_reorg))
        for row_start, row_end, blockr in zip(bands[:-1], bands[1:], range(self._pos_mean.shape[0])):
            for col_start, col_end, blockc in zip(bands[:-1], bands[1:], range(self._pos_mean.shape[1])):
                self._pos_blockmean[row_start:row_end, col_start:col_end] = self._pos_mean[blockr, blockc]
        # negative
        self._neg_blockmean = np.zeros(np.shape(self.edge_thresh_reorg))
        for row_start, row_end, blockr in zip(bands[:-1], bands[1:], range(self._neg_mean.shape[0])):
            for col_start, col_end, blockc in zip(bands[:-1], bands[1:], range(self._neg_mean.shape[1])):
                self._neg_blockmean[row_start:row_end, col_start:col_end] = self._neg_mean[blockr, blockc]
        
        self._blockmean = np.zeros(np.shape(self.edge_thresh_reorg))
        for row_start, row_end, blockr in zip(bands[:-1], bands[1:], range(self._mean.shape[0])):
            for col_start, col_end, blockc in zip(bands[:-1], bands[1:], range(self._mean.shape[1])):
                self._blockmean[row_start:row_end, col_start:col_end] = self._mean[blockr, blockc]

        self._masked_posmean = self._pos_blockmask * self._pos_blockmean
        self._masked_negmean = self._neg_blockmask * self._neg_blockmean 
        
        self._masked_combomean = self._masked_posmean + self._masked_negmean
        # self._masked_combomean = self._blockmask * self._blockmean
        
        self._masked_edgethresh = self.edge_thresh_reorg * (self._pos_blockmask + self._neg_blockmask)
        # self._masked_edgethresh = self.edge_thresh_reorg * self._blockmask

        if self.term_name == 'nodal':
            if rot_override is not None:
                rot = rot_override
            else:
                rot=False
            
            self.blockfig, self.blockax, self.cbar = pf.map_mat_braced(self.edge_thresh_reorg, 
                                                                       self.term_name, rot, 
                                                                       bands, names,
                                                                       **kwargs)
            plt.axis('off')
            plt.savefig(f'{outd}/{self.model_name}_{self.term_name}_thresh.svg')
            
            self.blockfig_t, self.blockax_t, self.cbar_t = pf.map_mat_braced(self._masked_combomean,
                                                                             self.term_name, rot, 
                                                                             bands, names,
                                                                             **kwargs)
            plt.axis('off')
            plt.savefig(f'{outd}/{self.model_name}_{self.term_name}_thresh_block.svg')


        elif self.term_name == 'edge':
            if rot_override is not None:
                rot = rot_override
            else:
                rot = True
            
            ut = np.triu(self._masked_combomean, 0)
            lt = np.tril(self.edge_thresh_reorg, -1)
            self.blockmat = ut + lt
            
            self.blockfig, self.blockax, self.cbar= pf.map_mat_braced(self.blockmat,
                                                                      self.term_name, rot, 
                                                                      bands, names,
                                                                      **kwargs)
            plt.axis('off')
            plt.savefig(f'{outd}/{self.model_name}_{self.term_name}.svg')

        
class Groupmodel:
    def __init__(self, beh, sublist_path, root):
        self.beh = beh
        if isinstance(sublist_path, (str, Path)):
            with open(sublist_path, 'r') as f:
                subs = f.readlines()
                subs = [i.strip() for i in subs]
                subs = subs[1:]
        elif isinstance(sublist_path, Iterable):
            subs = sublist_path
        else:
            raise ValueError('sublist_path must be a path to a text file containing subject ids or an iterable of subject ids')

        self.subs = subs
        self.r2 = []
        self.r2_adj = []
        self._i = []
        self._j = []
        self._ets = []
        self._i_pval = []
        self._j_pval = []
        self._ets_pval = []
        self.Fstat = []
        self.autocorr = []
        
        for sub in self.subs:
            with open(f'{root}/{sub}_fit.pkl', 'rb') as f:
                model = pickle.load(f)
            self.r2.append(model['rsq'])
            self.r2_adj.append(model['adj_rsq'])
            self._i.append([i['i'] for i in model['betas']])
            self._j.append([i['j'] for i in model['betas']])
            self._ets.append([i['ets'] for i in model['betas']])
            self._i_pval.append([i['i'] for i in model['pvalues']])
            self._j_pval.append([i['j'] for i in model['pvalues']])
            self._ets_pval.append([i['ets'] for i in model['pvalues']])
            self.Fstat.append(model['fstat'])
            self.autocorr.append(model['autocorr_params'])

        self.r2 = np.array(self.r2)
        self.r2_adj = np.array(self.r2_adj)
        self._i = np.array(self._i)
        self._j = np.array(self._j)
        self._ets = np.array(self._ets)
        self._i_pval = np.array(self._i_pval)
        self._j_pval = np.array(self._j_pval)
        self._ets_pval = np.array(self._ets_pval)
        self.Fstat = np.array(self.Fstat)
        self.autocorr = np.array(self.autocorr)
        
    def generate_term(self, ntotal):
        nodal = []
        nodal_p = []

        for ind in range(self._i.shape[0]):
            _n = mm.nodal_mat_reshape(self._i[ind, :], self._j[ind, :], ntotal)
            _np = mm.nodal_mat_reshape(self._i_pval[ind,:], self._j_pval[ind,:], ntotal)
            nodal.append(_n.flatten())
            nodal_p.append(_np.flatten())

        nodal = np.array(nodal)
        nodal_p = np.array(nodal_p)

        self.nodal = Modelterm(self.beh, 'nodal', nodal, nodal_p)
        self.edge = Modelterm(self.beh, 'edge', self._ets, self._ets_pval)
    
    def create_edge_modfit(self, imat, jmat, ijmat):
        isx = {}
        #1 Ni_Nj_Sij
        temp = ijmat - imat - jmat
        isx['Ni_Nj_Sij'] = np.where(temp == 1, 1, 0)
        #2 Ni_Sj_Sij or Si_Nj_Sij    
        temp1 = np.where((imat + jmat)==1, 1, 0)
        temp2 = ijmat + temp1 
        isx['Sn_Sij'] = np.where(temp2 == 2, 2, 0)
        #3 Si_Sj_Sij
        temp = ijmat + imat + jmat
        isx['Si_Sj_Sij'] = np.where(temp == 3, 3, 0)
        #3 Si_Nj_Nij or Ni_Si_Nij
        temp1 = np.where((imat + jmat)==1, 1, 0)
        temp2 = temp1 - ijmat 
        isx['Sn_Nij'] = np.where(temp2 == 1, 4, 0)
        #5 Si_Sj_Nij
        temp = imat + jmat - ijmat
        isx['Si_Sj_Nij'] = np.where(temp == 2, 5, 0)
        
        combo = np.zeros(np.shape(imat))
        for k, v in isx.items():
            combo += v
        
        self.isx = isx
        self.combo = combo
        
        
    def spin_test_edge_modfit(self, input_combo, uval, iters, rerun, spin_ord, bounds, alpha=0.05, name='full'):
               
        if type(rerun) != bool:
            raise AttributeError('`rerun` must be True or False')
        
        spin_dir = f'data/null_spin_models/{self.beh}'    
        if not os.path.exists(spin_dir):
            os.makedirs(spin_dir)
        
        if rerun:
            
            self.isx_count, self.isx_null = mm.spin_test_isx(input_combo, spin_ord, bounds, uval=uval, iters=iters)
            np.save(f'{spin_dir}/isx_count_{self.beh}_{iters}_{name}.npy', self.isx_count)
            np.save(f'{spin_dir}/isx_null_{self.beh}_{iters}_{name}.npy', self.isx_null)
            
        else:
            self.isx_count = np.load(f'{spin_dir}/isx_count_{self.beh}_{iters}_{name}.npy', allow_pickle=True)
            self.isx_null = np.load(f'{spin_dir}/isx_null_{self.beh}_{iters}_{name}.npy', allow_pickle=True)

        self.isx_block = mm.check_null_isx(self.isx_count, self.isx_null, alpha=alpha)
    
    def plot_isx(self, input_combo, bands, names, split=True, **kwargs):
        
        outd = f'figs/matrix_plots/{self.beh}'    
        if not os.path.exists(outd):
            os.makedirs(outd)
              
        rot = True
        
        tr_thresh = self.isx_block * mm.get_toprank(self.isx_count)

        self.expand_block_isx = mm.expand_block(tr_thresh, bands, lambda x: x)()
        
        if split:
            self.expand_block_isx = np.triu(self.expand_block_isx, 0) + np.tril(input_combo, -1)

            # drop 0
        
        self.expand_block_isx[self.expand_block_isx==0] = np.nan

        self.blockfig_isx, self.blockax_isx, self.cbar_isx = pf.map_mat_braced(self.expand_block_isx,
                                                                               self.beh, rot, 
                                                                               bands, names,
                                                                               **kwargs)

        plt.axis('off')
        plt.savefig(f'{outd}/isx_model_split-{split}_{self.beh}.svg')