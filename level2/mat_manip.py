#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:32:13 2024

@author: robfren
"""
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib

ROOT = Path(__file__).resolve().parents[1]


class Matmanip:
    def __init__(self):
        self.spin_mat_full = np.load(ROOT / 'misc' / 'spin_schaefer_full.npy')
        labfile = ROOT / 'misc' / 'schaefer200_tian1_labels.txt'

        with open(labfile, 'r') as f:
            labs = f.readlines()

        self.net7 = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
        self.net17 = ['VisCent', 'VisPeri', 'SomMotA', 'SomMotB', 'DorsAttnA', 'DorsAttnB', 'SalVentAttnA', 'SalVentAttnB',
                 'LimbicB', 'LimbicA', 'ContA', 'ContB', 'ContC', 'DefaultA', 'DefaultB', 'DefaultC', 'TempPar']
        self.b1 = {
            'Visual': 0,
            'Somatomotor': 24,
            'Dorsal Attention': 58,
            'Ventral Attention': 80,
            'Limbic': 106,
            'Control': 120,
            'Default Mode': 157,
            'Subcortical': 200,
            'placeholder': 216}

        self.b2 = {
            'VisCent': 0,
            'VisPeri': 12,
            'SomMot A': 24,
            'SomMot B': 43,
            'DorsAttn A': 58,
            'DorsAttn B': 70,
            'VentAttn A': 80,
            'VentAttn B': 96,
            'Limbic OFC': 106,
            'Limbic TempPole': 112,
            'Cont A': 120,
            'Cont B': 136,
            'Cont C': 151,
            'Default A': 157,
            'Default B': 171,
            'Default C': 188,
            'TempPar': 194,
            'Subcortical': 200,
            'placeholder': 216}

        self.labs = [lab.strip()[11:] for lab in labs[:200]] + [lab.strip() for lab in labs[200:]]

        self.bands = np.array([i for i in self.b1.values()])
        self.bands17 = np.array([i for i in self.b2.values()])

        self.bnames = [i for i in self.b1.keys()]
        self.bnames17 = [i for i in self.b2.keys()]
        
        self.syscol7 = [
            (162,82,171),
            (120,154,192),
            (64,152,48),
            (222,102,252),
            (246,254,197),
            (238,186,66),
            (217,113,124),
            ]
        
        self.syscol17 = [
            (162,82,171),
            (162,82,171),
            (120,154,192),
            (120,154,192),
            (64,152,48),
            (64,152,48),
            (222,102,252),
            (222,102,252),
            (246,254,197),
            (246,254,197),
            (238,186,66),
            (238,186,66),
            (238,186,66),
            (217,113,124),
            (217,113,124),
            (217,113,124),
            (217,113,124),
            ]
        
        # generate saturation/value shift of 7 net colors
        adj_val = 0.1

        self.hsv = [
            (0.816, 0.520+adj_val, 0.670-adj_val),
            (0.816, 0.520, 0.670),
            (0.587, 0.375+adj_val, 0.752-adj_val),
            (0.587, 0.375, 0.752),
            (0.307, 0.684+adj_val, 0.596-adj_val),
            (0.307, 0.684, 0.596),
            (0.799, 0.595+adj_val, 0.988-adj_val),
            (0.799, 0.595, 0.988),
            (0.190, 0.224+adj_val, 0.996-adj_val),
            (0.190, 0.224, 0.996),
            (0.116, 0.722+adj_val, 0.933-adj_val),
            (0.116, 0.722, 0.933),
            (0.116, 0.722-adj_val, 0.933+0.05),
            (0.982, 0.479+(2*adj_val), 0.850-(2*adj_val)),
            (0.982, 0.479+adj_val, 0.850-adj_val),
            (0.982, 0.479, 0.850),
            (0.982, 0.479-adj_val, 0.850),
            ]
        self.shifted_syscol17 = [tuple(matplotlib.colors.hsv_to_rgb(col)) for col in self.hsv]
        self.syscol17_lit = [
            (123,20,137),
            (225,42,51),
            (74,131,182),
            (67,200,164),
            (73,160,60),
            (2,121,15),
            (164,68,180),
            (235,155,198),
            (220,244,170),
            (122,136,50),
            (233,149,32),
            (135,50,74),
            (121,142,178),
            (237,227,48),
            (204,63,79),
            (56,38,135),
            (4, 44, 254),
            ]
        nets = []
        idx = []

        for net in self.net17:
            subnet = []
            subidx = []
            for lab in labs:
                if net in lab:
                    subnet.append(lab)
                    subidx.append(labs.index(lab))
            nets.append(subnet)
            idx.append(subidx)

        reorg_labs = sum(nets, [])
        reorg_idx = sum(idx, [])
        self.reorg_labs = reorg_labs + labs[200:]
        self.reorg_idx = reorg_idx + [i for i in range(200, len(labs))]

def reorg(mat, reorg_idx, trim=200):
    
    org_col = mat[:, reorg_idx]
    FC = org_col[reorg_idx, :]
    np.fill_diagonal(FC, 0)
    if trim == 'None':
        return FC
    else:
        trimmed = FC[:trim, :trim]
        return trimmed


def mask_based_utri2mat(utri, ntotal):
    """
    Convert vectorized upper triangle into a symmetric matrix of shape
    (ntotal, ntotal)

    Parameters
    ----------
    utri : np.1darray
        input vectorized upper triangle to convert.
    ntotal : int
        size of square matrix.

    Returns
    -------
    out : np.2darray
        output 2d symmetric matrix.
    """

    out = np.empty((ntotal, ntotal))
    mask = np.triu(np.ones((ntotal, ntotal), dtype=bool), k=1)
    out[mask] = utri
    out.T[mask] = utri
    
    return out
    
def nodal_mat_reshape(iflat, jflat, ntotal):
    """
    Convert nodal upper triangle vectors into nodal matrix outputs of shape
    (ntotal, ntotal).
    
    Parameters
    ----------
    iflat : np.1darray
        first input vectorized nodal upper tri.
    jflat : np.1darray
        second input vectorized nodal upper tri.
    ntotal : int
        size of square matrix.

    Returns
    -------
    nodal : np.2darray
        output 2d matrix, non-symmetric.
    """
    
    i = mask_based_utri2mat(iflat, ntotal)
    j = mask_based_utri2mat(jflat, ntotal)
    nodal = np.triu(i, 1) + np.tril(j.T, -1)
    
    return nodal

#TODO modify this to work using reduce mat
def counter(mat, bands, axis, method='count'):
    blocks = []

    # this loop pulls a subsection of the matrix sliced by the boundaries of the 'bands argument
    # it takes the block subsection and appends it to a list named 'blocks'
    # this makes the blocks easier to work with later.
    
    if axis == 'single':
        for row_start, row_end in zip(bands[:-1], bands[1:]):
            block = mat[row_start:row_end, :]
            blocks.append(block)
        
    elif axis == 'double':
    
        for row_start, row_end in zip(bands[:-1], bands[1:]):
            for col_start, col_end in zip(bands[:-1], bands[1:]):
                block = mat[row_start:row_end, col_start:col_end]
                blocks.append(block)
    else:
        raise ValueError('axis argument should be either "single" or "double"')
        
    conmat = []

    # this loops through the now separated blocks and uses np.where to assign above threshold values as True and
    # below False It then counts the Trues and appends that to 'conmat', 'conmat' is then reshaped into a square matrix
    for block in blocks:
        if method == 'count':
            bin_block = np.where(block != 0, True, False)
            unique, counts = np.unique(bin_block, return_counts=True)
            d = dict(zip(unique, counts))
            try:
                conmat.append(d[True])
            except KeyError:
                conmat.append(0)
        elif method == 'mean':
            conmat.append(np.mean(block))
        else:
            raise ValueError('method argument must be "count" or "mean"')

    conmat = np.reshape(np.array(conmat), (-1, len(bands) - 1))

    return conmat


def spin_test(in_mat, spin_ord, net_bounds, axis, iters=1000, method='count'):
    if iters > spin_ord.shape[1]:
        raise IndexError('Number of iterations out of range of input spin matrix')
    
    # create copy of the spin order to shuffle
    # shuffling the array allows for random selection of the first (shuffled) iters of spin_ord
    spin_copy = spin_ord[:, :].T
    np.random.shuffle(spin_copy)

    # create null model
    null_model = []
    for i in range(iters):
        # pull random permutation order, -1 to allow for 0 based index
        perm = spin_copy[i, :] - 1

        # rearrange according to indices
        if axis == 'single':
            spun = in_mat[perm, :]
        elif axis == 'double':
            temp = in_mat[:, perm]
            spun = temp[perm, :]
        else:
            raise ValueError('axis argument should be either "single" or "double"')

        # now get count or mean w/in block
        null_model.append(counter(spun, net_bounds, axis, method))

    null_model = np.array(null_model)

    true_blocks = counter(in_mat, net_bounds, axis, method)

    return [true_blocks, null_model]

def count_isx(block, unique_vals):
    counts = {}
    for val in unique_vals:
        temp = np.where(block==val, 1, 0)
        counts[val] = np.sum(temp)
    return counts
	
	
def spin_test_isx(in_mat, spin_ord, net_bounds, uval, iters=1000):
    if iters > spin_ord.shape[1]:
        raise IndexError('Number of iterations out of range of input spin matrix')

    # create copy of the spin order to shuffle
    # shuffling the array allows for random selection of the first (shuffled) iters of spin_ord
    spin_copy = spin_ord[:, :].T
    np.random.shuffle(spin_copy)
    
    # create null model
    null_model = []
    for i in range(iters):
        # pull random permutation order, -1 to allow for 0 based index
        perm = spin_copy[i, :] - 1
        
        # rearrange according to indices
        temp = in_mat[:, perm]
        spun = temp[perm, :]
        
        # now get count or mean w/in block
        null_model.append(reduce_block(spun, net_bounds, count_isx, dtype='O')(unique_vals=uval))
        
    null_model = np.array(null_model)
    
    true_blocks = reduce_block(in_mat, net_bounds, count_isx, dtype='O')(unique_vals=uval)
    
    return [true_blocks, null_model]

def check_null_isx(model, null_model, alpha=0.05):
    
    sig_block = np.zeros(np.shape(model))
    for rowi in range(model.shape[0]):
        for coli in range(model.shape[1]):
            t_block = model[rowi, coli]
            keys = [int(k) for k, _ in t_block.items()]
            vals = [int(v) for _, v in t_block.items()]
            toprank = keys[vals.index(max(vals))]
            null_full = null_model[:, rowi, coli]
            null = np.array([int(d[toprank]) for d in null_full])
            
            supra = np.where(null < t_block[toprank], 1, 0)
            if (np.sum(supra) / null.shape[0]) > (1 - alpha):
                sig_block[rowi, coli] = 1

    return sig_block

def get_toprank(isx_count):
    toprank = np.zeros(np.shape(isx_count))
    for rowi in range(toprank.shape[0]):
        for coli in range(toprank.shape[1]):
            t_block = isx_count[rowi, coli]
            keys = [int(k) for k, v in t_block.items()]
            vals = [int(v) for k, v in t_block.items()]
            tr = keys[vals.index(max(vals))]
            toprank[rowi, coli] = tr
    
    return toprank

    
def check_null(model, null_model, axis, alpha=0.05):
    sig_block = np.zeros(np.shape(model))
    sup_block = np.zeros(np.shape(model))
    
    if axis == 'single':
        for rowi in range(model.shape[1]):
            t_block = model[0, rowi]
            null = null_model[:, 0, rowi]
            supra = np.where(null < t_block, 1, 0)
            sup_block[0, rowi] = 1-(np.sum(supra) / null.shape[0])
        
        sup_fdr = np.reshape(stats.false_discovery_control(sup_block.flatten()), sup_block.shape)
        sig_block = np.where(sup_fdr < alpha, 1, 0)
                
    elif axis == 'double':
        for rowi in range(model.shape[0]):
            for coli in range(model.shape[1]):
                t_block = model[rowi, coli]
                null = null_model[:, rowi, coli]
                supra = np.where(null < t_block, 1, 0)
                sup_block[rowi, coli] = 1-(np.sum(supra) / null.shape[0])
            
        sup_fdr = np.reshape(stats.false_discovery_control(sup_block.flatten()), sup_block.shape)
        sig_block = np.where(sup_fdr < alpha, 1, 0)
        
    else:
        raise ValueError('axis argument should be either "single" or "double"')

    return sig_block

def nonzeromean(block):
    nanblock = np.where(block==0, np.nan, block)
    blockmean = np.nanmean(nanblock)
    if np.isnan(blockmean):
        blockmean = 0
    return blockmean

def reduce_block(mat, bands, func_to_iter, dtype='float'):
    def wrapper_red(*args, **kwargs):
        bands_phold = list(bands) + [int(bands[-1])+1]
        blockmat = np.empty((len(bands)-1, len(bands)-1), dtype=dtype)
        
        for row_start, row_end, blockr in zip(bands_phold[:-1], bands_phold[1:], range(blockmat.shape[0])):
            for col_start, col_end, blockc in zip(bands_phold[:-1], bands_phold[1:], range(blockmat.shape[1])):
                blockmat[blockr, blockc] = func_to_iter(mat[row_start:row_end, col_start:col_end], *args, **kwargs)
        return blockmat
    return wrapper_red

def expand_block(blockmat, bands, func_to_iter, dtype='float'):
    def wrapper_ex(*args, **kwargs):
        bands_phold = list(bands) + [int(bands[-1])+1]
        mat = np.empty((bands[-1], bands[-1]), dtype=dtype)
        
        for row_start, row_end, blockr in zip(bands_phold[:-1], bands_phold[1:], range(blockmat.shape[0])):
            for col_start, col_end, blockc in zip(bands_phold[:-1], bands_phold[1:], range(blockmat.shape[1])):
                mat[row_start:row_end, col_start:col_end] = func_to_iter(blockmat[blockr, blockc], *args, **kwargs)
        return mat
    return wrapper_ex


