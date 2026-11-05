#%%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:29:07 2024

@author: robfren
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import mat_manip as mm
import termpull as tp
import plotting_funcs as pf

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.plotting import (
    find_parcellation_cut_coords,
    plot_connectome, show,
)

import pickle

#%%

ROOT = Path(__file__).resolve().parents[1]


# SETTING GLOBALS

ya_sublist = ROOT / 'subject_lists' / 'ya_sublist_goodsubs.tsv'
oa_sublist = ROOT / 'subject_lists' / 'oa_sublist_goodsubs.tsv'
group_sublist = ROOT / 'subject_lists' / 'sublist_goodsubs.tsv'

miscvar = mm.Matmanip()

ntotal = 216
trim = 200
spin_iters = 10_000
thr = 0.05


datadir = ROOT / 'level1' / 'data' / 'sub_level_edge_model_prewhiten_awk'
rest_dir = ROOT / 'RSFC_control' / 'RSFC_Data'

beh = 'awk'

debug_surf_maps = {}

rerun_spin = True
include_rest = False
contour = True
contour_group = False
contour_col = "#00cf45ff"

load_models = True

if load_models == True:
    save_models = False
else:
    save_models = True




# run YA and OA unthresholded maps and create surface plots
models = {}

for grouplabel, sublist in zip(['YA', 'OA'], [ya_sublist, oa_sublist]):
    print('\n\n-------------------------------')
    print(f'starting {beh} {grouplabel}...'.upper())
    print('-------------------------------\n')

    awk_model = tp.Groupmodel(f'{beh}_{grouplabel}', sublist, datadir)
    awk_model.generate_term(ntotal)

    print('running t-tests and reorganizing matrices...')
    # edge level t-test
    awk_model.nodal.edge_sigtest(thresh_level=thr)
    awk_model.edge.edge_sigtest(thresh_level=thr)

    # run matrix cleaning - reorganize + trim subcortical - needed for later t test between beh
    awk_model.nodal.createmats(ntotal, miscvar.reorg_idx, trim)
    awk_model.edge.createmats(ntotal, miscvar.reorg_idx, trim)

    print('running spin tests...')
    awk_model.nodal.spin_test(iters=spin_iters, rerun=rerun_spin, spin_ord=miscvar.spin_mat_full, bounds=miscvar.bands17[:-1], alpha=thr)
    awk_model.edge.spin_test(iters=spin_iters, rerun=rerun_spin, spin_ord=miscvar.spin_mat_full, bounds=miscvar.bands17[:-1], alpha=thr)
    
    models[grouplabel] = awk_model

    print('plotting...')
    adj_PuOr_r = pf.adj_cmap(plt.cm.PuOr_r, exp=1.5)


# determine shared min and max for plotting
shared_min_node = np.nanmin([models['YA'].nodal.edge_thresh_reorg, models['OA'].nodal.edge_thresh_reorg])
shared_max_node = np.nanmax([models['YA'].nodal.edge_thresh_reorg, models['OA'].nodal.edge_thresh_reorg])
shared_node_bound = np.nanmax([np.abs(shared_min_node), np.abs(shared_max_node)])

shared_min_edge = np.nanmin([models['YA'].edge.edge_thresh_reorg, models['OA'].edge.edge_thresh_reorg])
shared_max_edge = np.nanmax([models['YA'].edge.edge_thresh_reorg, models['OA'].edge.edge_thresh_reorg])

for grouplabel in ['YA', 'OA']:
    awk_model = models[grouplabel]

    nilearn_kwargs = {'cmap': adj_PuOr_r, 'threshold': 0.001}

    # plot surface for node:
    if contour_group:
        contour_bin_org = np.where(np.nanmean(awk_model.nodal._masked_combomean, axis=1)[:200] !=0, 1, 0)
        contour_bin = contour_bin_org[np.argsort(miscvar.reorg_idx[:200])]
    else:
        contour_bin = np.zeros(200)

    surf_mat = np.nanmean(awk_model.nodal.edge_t_mat, axis=1)[:200]
    
    debug_surf_maps[f'{beh}_{grouplabel}'] = (surf_mat, contour_bin)



    pf.plot_surf(surf_mat, contour_bin, f'{beh}_contour-{contour_group}_{grouplabel}', 
                    contour_col = contour_col, nilearn_kwargs = nilearn_kwargs,
                    sym_cbar=True, mesh_qual='high', bval=shared_node_bound)

    os.makedirs(f'figs/matrix_plots/{beh}_{grouplabel}', exist_ok=True)

    # plot node unthresholded matrix
    blockfig, blockax, cbar = pf.map_mat_braced(awk_model.nodal.edge_t_reorg,
                                                f'node_{grouplabel}_unthresh', False, 
                                                miscvar.bands17[:-1], miscvar.bnames17[:-1],
                                                **{'cmap': adj_PuOr_r, 'vmin': -shared_node_bound, 'vmax': shared_node_bound})
    plt.axis('off')
    plt.savefig(f'figs/matrix_plots/{beh}_{grouplabel}/{beh}_{grouplabel}_node_unthresh.svg')
    
    # plot edge unthresholded matrix
    blockfig, blockax, cbar= pf.map_mat_braced(awk_model.edge.edge_t_reorg,
                                                f'edge_{grouplabel}_unthresh', False, 
                                                miscvar.bands17[:-1], miscvar.bnames17[:-1],
                                                **{'cmap': adj_PuOr_r, 'vmin': shared_min_edge, 'vmax': shared_max_edge})
    plt.axis('off')
    plt.savefig(f'figs/matrix_plots/{beh}_{grouplabel}/{beh}_{grouplabel}_edge_unthresh.svg')

    print(f'finished {beh} {grouplabel}')




# run age model with both YA and OA
if load_models:
    with open(f'figs/matrix_plots/{beh}_group_overall_effect_rest_cov-{include_rest}/overall_model.pkl', 'rb') as f:
        overall_model = pickle.load(f)
    with open(f'figs/matrix_plots/{beh}_group_age_effect_rest_cov-{include_rest}/age_model.pkl', 'rb') as f:
        age_model = pickle.load(f)
else:
    print('\n\n-------------------------------')
    print(f'running omnibus age test...'.upper())
    print('-------------------------------\n')

    overall_model = tp.Groupmodel(f'{beh}_group_overall_effect_rest_cov-{include_rest}', group_sublist, root)
    overall_model.generate_term(ntotal)
    age_model = tp.Groupmodel(f'{beh}_group_age_effect_rest_cov-{include_rest}', group_sublist, root)
    age_model.generate_term(ntotal)

    OA_idx = np.where([int(sub.split('-')[1]) < 1000 for sub in overall_model.subs])[0]
    YA_idx = np.where([int(sub.split('-')[1]) > 1000 for sub in overall_model.subs])[0]

    if include_rest:
        print('running age model with resting state as covariate...')
        allsub_rest_node = np.empty(age_model.nodal.sub_beta.shape)
        allsub_rest_edge = np.empty(age_model.edge.sub_beta.shape)

    # node shape (subjects, 46656)
    # edge shape (subjects, 23220)

    # steps 1. node: flatten OR edge: pull out upper tri and flatten 2, insert into allsub_rest_node or allsub_rest_edge

        for idx, sub_id in enumerate(overall_model.subs):
            xcp_mat = np.loadtxt(f'{rest_dir}/{sub_id}/{sub_id}_task-rest_space-MNI152NLin6Asym_res-2_atlas-Schaefer200Tian1_measure-pearsoncorrelation_conmat.tsv', delimiter='\t', dtype=str)
            xcp_mat = xcp_mat[1:, 1:].astype(float)

            allsub_rest_node[idx, :] = xcp_mat.flatten()
            allsub_rest_edge[idx, :] = xcp_mat[np.triu_indices(xcp_mat.shape[0], k=1)]


        allsub_rest_node_Z = np.arctanh(allsub_rest_node)
        allsub_rest_edge_Z = np.arctanh(allsub_rest_edge)
        allsub_rest_node_Z[np.isinf(allsub_rest_node_Z)] = 0


    for term in ['nodal', 'edge']:

        overall_term = getattr(overall_model, term)
        age_term = getattr(age_model, term)

        awk_sub_beta = getattr(overall_model, term).sub_beta

        ya_sub_b = awk_sub_beta[YA_idx,:]
        oa_sub_b = awk_sub_beta[OA_idx,:]

        if include_rest:
            if term == 'nodal':
                rest_cov = allsub_rest_node_Z
            else:
                rest_cov = allsub_rest_edge_Z   
            
            ya_rest = rest_cov[YA_idx,:]
            oa_rest = rest_cov[OA_idx,:]

        overall = []
        overall_pval = []
        age = []
        age_pval = []

        # TODO: vectorize this to speed up, but for now just run in a loop

        print(f'running t-tests for group {term}')
        for cell_idx in range(ya_sub_b.shape[1]):
            if cell_idx % 5000 == 0:
                print(f'working on cell {cell_idx} of {ya_sub_b.shape[1]}...')
            
            ya_b_cell = ya_sub_b[:, cell_idx]
            oa_b_cell = oa_sub_b[:, cell_idx]

            y = np.concatenate((ya_b_cell, oa_b_cell))

            if np.isnan(y).any():
                overall.append(np.nan)
                overall_pval.append(np.nan)
                age.append(np.nan)
                age_pval.append(np.nan)
            
            else:
                group = np.array([0.5]*len(ya_b_cell) + [-0.5]*len(oa_b_cell))
                if include_rest:
                    rest = np.concatenate((ya_rest[:, cell_idx], oa_rest[:, cell_idx]))
                    X = np.column_stack([np.ones(group.shape), group, rest])
                else:
                    X = np.column_stack([np.ones(group.shape), group])

                model = sm.OLS(y, X).fit(cov_type='HC3', use_t=True)
                
                overall.append(model.tvalues[0])
                overall_pval.append(model.pvalues[0])
                age.append(model.tvalues[1])
                age_pval.append(model.pvalues[1])

        overall_term.edge_t = np.array(overall)
        overall_term.edge_sig = np.array(overall_pval)
        age_term.edge_t = np.array(age)
        age_term.edge_sig = np.array(age_pval)

        # need to FDR correct, then threshold
        for mod_term in [overall_term, age_term]:
            mod_term.edge_sig[np.isnan(mod_term.edge_sig)] = 1
            mod_term.edge_t[np.isnan(mod_term.edge_t)] = 0

            mod_term.edge_sig = stats.false_discovery_control(mod_term.edge_sig)

            mod_term.edge_sig_bin = np.where(mod_term.edge_sig < 0.05, 1, 0)
            mod_term.edge_thresh = np.multiply(mod_term.edge_sig_bin, mod_term.edge_t)
        
        print(f'running spin tests for group {term} effects...')
        overall_term.createmats(ntotal, miscvar.reorg_idx, trim)
        age_term.createmats(ntotal, miscvar.reorg_idx, trim)

        # spin test and plot
        overall_term.spin_test(iters=spin_iters, rerun=rerun_spin, spin_ord=miscvar.spin_mat_full, bounds=miscvar.bands17[:-1], alpha=thr)
        age_term.spin_test(iters=spin_iters, rerun=rerun_spin, spin_ord=miscvar.spin_mat_full, bounds=miscvar.bands17[:-1], alpha=thr)


#%% save and load models for ease later

if save_models:
    with open(f'figs/matrix_plots/{beh}_group_overall_effect_rest_cov-{include_rest}/overall_model.pkl', 'wb') as f:
        pickle.dump(overall_model, f)
    with open(f'figs/matrix_plots/{beh}_group_age_effect_rest_cov-{include_rest}/age_model.pkl', 'wb') as f:
        pickle.dump(age_model, f)




#%%


adj_RdBu_r = pf.adj_cmap(plt.cm.RdBu_r, exp=1.5)
nilearn_kwargs = {'cmap': adj_RdBu_r, 'threshold': 0.0001}
 

overall_bound = np.nanmax([np.abs(np.nanmin(overall_model.nodal.edge_thresh_reorg)), np.abs(np.nanmax(overall_model.nodal.edge_thresh_reorg))])
age_bound = np.nanmax([np.abs(np.nanmin(age_model.nodal.edge_thresh_reorg)), np.abs(np.nanmax(age_model.nodal.edge_thresh_reorg))])

overall_kwargs = {'cmap': adj_PuOr_r,
                'vmin': -overall_bound, 
                'vmax': overall_bound}

overall_model.nodal.plot_block(miscvar.bands17[:-1], miscvar.bnames17[:-1], **overall_kwargs)
overall_model.edge.plot_block(miscvar.bands17[:-1], miscvar.bnames17[:-1], **overall_kwargs, rot_override=False)

nodal_kwargs = {'cmap': adj_RdBu_r,
                'vmin': -age_bound, 
                'vmax': age_bound}
edge_kwargs = {'cmap': adj_RdBu_r}

age_model.nodal.plot_block(miscvar.bands17[:-1], miscvar.bnames17[:-1], **nodal_kwargs)
age_model.edge.plot_block(miscvar.bands17[:-1], miscvar.bnames17[:-1], **edge_kwargs, rot_override=False)


#%%

if contour:
    contour_bin_org = np.where(np.nanmean(overall_model.nodal._masked_combomean, axis=1)[:200] !=0, 1, 0)
    contour_bin = contour_bin_org[np.argsort(miscvar.reorg_idx[:200])]
else:
    contour_bin = np.zeros(200)

surf_mat = np.nanmean(overall_model.nodal.edge_t_mat, axis=1)[:200]

pf.plot_surf(surf_mat, contour_bin, f'{beh}_contour-{contour}_group_overall_effect_rest_cov-{include_rest}',
                contour_col = contour_col, nilearn_kwargs = {'cmap': adj_PuOr_r, 'threshold': 0.0001}, 
                sym_cbar=True, mesh_qual='high', bval=overall_bound, border_rings=2)

if contour:
    contour_bin_org = np.where(np.nanmean(age_model.nodal._masked_combomean, axis=1)[:200] !=0, 1, 0)
    contour_bin = contour_bin_org[np.argsort(miscvar.reorg_idx[:200])]
else:
    contour_bin = np.zeros(200)

surf_mat = np.nanmean(age_model.nodal.edge_t_mat, axis=1)[:200]

pf.plot_surf(surf_mat, contour_bin, f'{beh}_contour-{contour}_group_age_effect_rest_cov-{include_rest}',
                contour_col = contour_col, nilearn_kwargs = nilearn_kwargs, 
                sym_cbar=True, mesh_qual='high', bval=age_bound, border_rings=2)


#%% pull all edges for all subjects on sig systems (node/edge), plot by age


systems = miscvar.bnames17[:-1]
bounds = [(lb, ub) for lb, ub in zip(miscvar.bands17[:-1], miscvar.bands17[1:])]
sys_lookup = {sys: bounds for sys, bounds in zip(systems, bounds)}
# allows for a call like sys_lookup['DAN_A'] to return the lower and upper bounds for that system.


#%%

def get_sys_edges(modelterm, systema, systemb=None):
    lb, ub = sys_lookup[systema]
    lb2, ub2 = 0, -1
    if systemb is not None:
        lb2, ub2 = sys_lookup[systemb]
    
    # mask for significant edges only
    sig_edges = np.where(modelterm.edge_thresh_reorg !=0, 1, 0)
    masked_beta = np.where(sig_edges[None, :, :], modelterm.sub_beta_mat_reorg, np.nan)
    ya_sys = masked_beta[YA_idx, lb:ub, lb2:ub2]
    oa_sys = masked_beta[OA_idx, lb:ub, lb2:ub2]
    ya_sys_means = np.nanmean(ya_sys, axis=(1, 2))
    oa_sys_means = np.nanmean(oa_sys, axis=(1, 2))  

    return ya_sys_means, oa_sys_means

def plot_boxplot(system_tuples, modelterm, savepath, YA_col="#e24646", OA_col="#424de2", figsize=(10, 8)):
    betas = []
    ages = []
    systems = []

    for systema, systemb in system_tuples:
        ya_sys_means, oa_sys_means = get_sys_edges(modelterm, systema, systemb)

        sys_name = f"{systema} - {systemb}" if systemb is not None else f"{systema}"

        ya_vals = ya_sys_means.ravel()
        oa_vals = oa_sys_means.ravel()

        vals = np.concatenate([ya_vals, oa_vals])

        betas.append(vals)
        ages.append(np.repeat(["YA", "OA"], [ya_vals.size, oa_vals.size]))
        systems.append(np.repeat(sys_name, vals.size))

    sys_df = pd.DataFrame({
        "Betas": np.concatenate(betas),
        "Age": np.concatenate(ages),
        "System": np.concatenate(systems),
    })

    plt.figure(figsize=figsize)

    sns.boxplot(
        data=sys_df,
        x="System",
        y="Betas",
        hue="Age",
        palette={"YA": YA_col, "OA": OA_col}
    )

    plt.axhline(0, color="black", linestyle="--", alpha=0.7)
    sns.despine()
    plt.tight_layout()
    os.makedirs(savepath, exist_ok=True)
    plt.savefig(f"{savepath}/{modelterm.term_name}_sig_systems_boxplot.svg")

#%%

node_tuples = [
    ('Cont C',None), 
]

edge_tuples = [
    ('SomMot B', 'DorsAttn A'),
    ('SomMot B', 'DorsAttn B'),
    ('SomMot B', 'Cont A'),
]

plot_boxplot(node_tuples, age_model.nodal, f'figs/matrix_plots/{beh}_group_age_effect_rest_cov-{include_rest}/node_sig_systems_boxplot', figsize=(4,7))
plot_boxplot(edge_tuples, age_model.edge, f'figs/matrix_plots/{beh}_group_age_effect_rest_cov-{include_rest}/edge_sig_systems_boxplot', figsize=(9,7))

#%% create edge plot of sig edges

sys17norm = [(r/255,g/255,b/255) for r,g,b in miscvar.syscol17_lit]

def plot_sig_edges(systems, input_model, savepath, filename=None):
    
    schaefer = fetch_atlas_schaefer_2018(n_rois=trim, yeo_networks=17)
    coordinates = find_parcellation_cut_coords(labels_img=schaefer["maps"])
    coordinates = coordinates[miscvar.reorg_idx[:200], :]
    
    sysmask = np.zeros((200,200))
    for systema, systemb in systems:
        a_lb, a_ub = sys_lookup[systema]
        b_lb, b_ub = sys_lookup[systemb]

        sysmask[a_lb:a_ub, b_lb:b_ub] = 1
        sysmask[b_lb:b_ub, a_lb:a_ub] = 1

    mat = input_model.edge.edge_thresh_reorg * sysmask
    mat = np.where(mat==0, np.nan, mat)

    ax_mean = np.nanmean(mat, axis=0)
    valid_node_idx = np.argwhere(~np.isnan(ax_mean))
    valid_node_idx = valid_node_idx.flatten()

    node_colors = []
    for length, color in zip(np.diff(miscvar.bands17[:-1]), sys17norm):
        cur_col = [color]*length
        node_colors += cur_col

    # index node coordinates and node colors

    rem_coords = coordinates[valid_node_idx, :]
    rem_colors = [node_colors[idx] for idx in valid_node_idx]

    # drop rows/columns of all nans in matrix
    org_col = mat[:, valid_node_idx]
    rem_mat = org_col[valid_node_idx, :]

    os.makedirs(savepath, exist_ok=True)
    sys_list = [f"{sysa.replace(' ', '').upper()}-{sysb.replace(' ', '').upper()}" for sysa, sysb in systems]
    if len(sys_list) <= 2:
        sysconcat = "_".join(sys_list)
    else:
        sysconcat = 'MANYSYS'

    if filename is None:
        filename = f'edge_graph_{input_model.beh}_{sysconcat}'

    connectome2d = plot_connectome(
        adjacency_matrix = rem_mat, 
        node_coords = rem_coords, 
        node_color = rem_colors,
        node_size = 25,
        colorbar = True,
        edge_cmap=adj_RdBu_r,
        edge_threshold=None,
        edge_kwargs = {'alpha': 0.7},
        output_file = f'{savepath}/{filename}.svg'
    )

    show()

#%%
edge_tuples = [
    ('SomMot B', 'DorsAttn A'),
    ('SomMot B', 'DorsAttn B'),
    ('SomMot B', 'Cont A'),
]

plot_sig_edges(edge_tuples, age_model, 'figs/brain_graphs')

#%%

plot_sig_edges([edge_tuples[0]], age_model, 'figs/brain_graphs')
plot_sig_edges([edge_tuples[1]], age_model, 'figs/brain_graphs')
plot_sig_edges([edge_tuples[2]], age_model, 'figs/brain_graphs')



# %% create schaefer parcel legend on surface

surf_mat = np.empty(200)

for index, (lb, ub) in enumerate(bounds):
    index+=1
    surf_mat[lb:ub] = index


idx = miscvar.reorg_idx[:200]

# surf_mat is currently in reordered order
surf_mat_original_order = np.empty_like(surf_mat)

# undo the reordering
surf_mat_original_order[idx] = surf_mat

custom_schaefer17 = ListedColormap(sys17norm)

contour_bin = np.ones(200)

pf.plot_surf(surf_mat_original_order, contour_bin, f'schaefer_parcel_legend',
            contour_col = 'black', nilearn_kwargs = {'cmap': custom_schaefer17, 'threshold': 0.0001}, 
            vmin=0.5, vmax=17.5, sym_cbar=False, mesh_qual='high')



#%%

mat = overall_model.edge.pos_block + overall_model.edge.neg_block
mean = overall_model.edge._pos_mean + overall_model.edge._neg_mean
mat = np.triu(mat)
idx = np.argwhere(mat != 0)

blocknames = []
blockvals = []
for i,j in idx:
    blocknames.append(f'{miscvar.bnames17[i]} - {miscvar.bnames17[j]}')
    blockvals.append(mean[i,j])



df = pd.DataFrame()
df['system names'] = blocknames
df['mean t-statistic'] = blockvals


df.to_csv(f'figs/matrix_plots/{beh}_group_overall_effect_rest_cov-{include_rest}/sig_systems_mean_tstats.csv', index=False)


# %%
