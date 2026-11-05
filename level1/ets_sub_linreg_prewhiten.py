import sys
from pathlib import Path
import os
import time
import numpy as np
import scipy as sp
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import pickle

# %%

ROOT = Path(__file__).resolve().parents[1]


bdir = sys.argv[1] # point to directory containing subject data

subj = sys.argv[2] # subject ID
beh = sys.argv[3] # behavior label
beh_rating = sys.argv[4] # consensus vs individual flag
if beh_rating != 'consensus' and beh_rating != 'individual':
    raise ValueError("beh_rating argument must be either 'consensus' or 'individual'")

indices = np.load('./indices.npy')

# Path to control regressor
if len(sys.argv) > 4:
    AddReg = sys.argv[5]
else:
    AddReg = None

# %%
debug = True
FullRes = False # save full statsmodels result output instead of only necessary params

if debug:
    start_time = time.time()

shop_ds_len = 249
zoo_ds_len = 233

# %% read in subject nodal timeseries

zoo_nodal = np.load(f'{bdir}/{subj}/{subj}_pettingzoo_interpolated_ts_custom_parc.npy')
shop_nodal = np.load(f'{bdir}/{subj}/{subj}_souvenirshop_interpolated_ts_custom_parc.npy')
zoo_nodal = zoo_nodal[:zoo_ds_len, :]
shop_nodal = shop_nodal[:shop_ds_len, :]
combo_nodal = np.append(zoo_nodal, shop_nodal, axis=0)

if beh_rating == 'consensus':     
    y = np.load(ROOT / 'regs' / 'hrf' / f'{beh}_hrf.npy')  # behavior (dependent variable)
elif beh_rating == 'individual':
    y = np.load(ROOT / 'regs' / 'hrf' / f'{subj}_{beh}_hrf.npy')

y = np.append(y[:zoo_ds_len], y[zoo_ds_len:zoo_ds_len + shop_ds_len])
y = np.reshape(y, (len(y), 1))
dm = np.array([1 for t in range(zoo_ds_len)] + [0 for t in range(shop_ds_len)])  # dummy coded run variable

if AddReg is not None:
    # additional inclusion of a regressor confound
    reg = np.load(AddReg)
    reg = np.append(reg[:zoo_ds_len], reg[zoo_ds_len:zoo_ds_len + shop_ds_len])

# %%

combo_i = []
combo_j = []
combo_ij = []

for edge in range(int((combo_nodal.shape[1] - 1) * combo_nodal.shape[1] / 2)):
    # get indices for nodes
    r, c = indices[edge].split('-')
    i = combo_nodal[:, int(r)]  # node i timeseries
    j = combo_nodal[:, int(c)]  # node j timeseries
    
    # standardize timeseries
    i_std = (i - i.mean()) / i.std()
    j_std = (j - j.mean()) / j.std()
    
    # edge timeseries
    ij = i_std * j_std
    
    # append to larger matrix timeseries for later
    combo_i.append(i_std)
    combo_j.append(j_std)
    combo_ij.append(ij)

combo_i = np.array(combo_i).T
combo_j = np.array(combo_j).T
combo_ij = np.array(combo_ij).T


nonnan_idx = np.argwhere(~np.isnan(y))[:,0]
y = y[nonnan_idx,:]
combo_i = combo_i[nonnan_idx, :]
combo_j = combo_j[nonnan_idx, :]
combo_ij = combo_ij[nonnan_idx, :]
dm = dm[nonnan_idx]

if AddReg is not None:
	reg = reg[nonnan_idx]

# %%

e = np.zeros(combo_ij.shape)

for edge in range(combo_ij.shape[1]):
    i = combo_i[:, edge]  # node i timeseries
    j = combo_j[:, edge]  # node j timeseries
    ij = combo_ij[:, edge]  # edge timeseries (interaction)
    
    if AddReg is not None:
        X = np.array([i, j, ij, dm, reg, np.ones(ij.shape[0])]).T
    else:
        X = np.array([i, j, ij, dm, np.ones(ij.shape[0])]).T
    # beta weights
    B = np.linalg.inv(X.T @ X) @ X.T @ y
    # get residuals
    e[:, edge] = np.squeeze(y - (X @ B))

# %% fit autoregressive model

# NOTE: AutoReg model in statsmodels here uses a conditional Max. Likelihood Estimator while MatLab AR model uses
#       Least-Squares Estimator

p = 25  # order for ar models
nsample = 100  # number of random edge samples to pull to fit models

P = []
for edge in np.random.choice(e.shape[1], nsample, replace=False):
    res = AutoReg(e[:, edge], p).fit()
    P.append(res.params)

P = np.array(P)
mu = np.nanmean(P, 0)

# %% construct V inverse to get whitening matrix

Amat = np.zeros((e.shape[0], e.shape[0]))

i = 1
for m in mu:
    vec = np.ones(e.shape[0] - i)
    d = np.diag(-m * vec, -i)
    Amat += d
    i += 1

Vinv = Amat @ Amat.T

U, S, _ = np.linalg.svd(Vinv)

# Compute whitening matrix
W = U @ sp.linalg.sqrtm(np.diag(S)) @ U.T

# %% multiply whitening matrix by all variables

Wy = W[p:-p, p:-p] @ y[p:-p]

Wi = W[p:-p, p:-p] @ combo_i[p:-p, :]
Wj = W[p:-p, p:-p] @ combo_j[p:-p, :]
Wij = W[p:-p, p:-p] @ combo_ij[p:-p, :]
Wdm = W[p:-p, p:-p] @ dm[p:-p]
if AddReg is not None:
    Wreg = W[p:-p, p:-p] @ reg[p:-p]

# set up structures to hold results
fullres = []  # full fit OLS model
fval = []  # fvalue of model
fprob = []  # p value of model
beta = []  # beta weights for each variable (nodal activity + ets)
sig = []  # p values for each variable (nodal activity + ets)
r2 = []  # model r2 to assess fit
r2_adj = []

# rerun regression with prewhitened data
for edge in range(Wij.shape[1]):
    if AddReg is not None:
        X = np.array([Wij[:, edge], Wi[:, edge], Wj[:, edge], Wdm, Wreg]).T
    else:
        X = np.array([Wij[:, edge], Wi[:, edge], Wj[:, edge], Wdm]).T
    X = sm.add_constant(X)
    res = sm.OLS(Wy, X).fit()

    betas = {}
    sigs = {}

    fullres.append(res)
    fval.append(res.fvalue)
    fprob.append(res.f_pvalue)
    betas['ets'] = res.params[1]
    betas['i'] = res.params[2]
    betas['j'] = res.params[3]
    beta.append(betas)

    sigs['ets'] = res.pvalues[1]
    sigs['i'] = res.pvalues[2]
    sigs['j'] = res.pvalues[3]
    sig.append(sigs)
    r2_adj.append(res.rsquared_adj)
    r2.append(res.rsquared)


# get final runtime for single subject
if debug:
    end_time = time.time()
    runtime = end_time - start_time
    print(f'runtime(s):\n\t{runtime}')


if AddReg is not None:
    reglab = f'_{AddReg.split("/")[-1][:-4]}'
else:
    reglab = ''

os.makedirs(ROOT / 'level1' / 'data' / f'sub_level_edge_model_prewhiten_{beh_rating}_{beh}{reglab}', exist_ok=True)

if FullRes:
	with open(ROOT / 'level1' / 'data' / f'sub_level_edge_model_prewhiten_{beh_rating}_{beh}{reglab}' / f'{subj}_full_model.pkl', 'wb') as f:
		pickle.dump(fullres, f)

fit_d = {
    'fstat': fval,
    'fprob': fprob,
    'betas': beta,
    'pvalues': sig,
    'adj_rsq': r2_adj,
    'rsq': r2,
    'autocorr_params': mu,
}

with open(ROOT / 'level1' / 'data' / f'sub_level_edge_model_prewhiten_{beh_rating}_{beh}{reglab}' / f'{subj}_fit.pkl', 'wb') as f:
    pickle.dump(fit_d, f)

