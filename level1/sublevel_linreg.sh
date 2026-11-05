#!/bin/bash

module load conda
conda activate awkwardness_ets

datadir=$1
subj=$2
regname=$3
AddReg=$4


python ets_sub_linreg_prewhiten_consensus.py ${datadir} ${subj} ${regname} ${AddReg}

