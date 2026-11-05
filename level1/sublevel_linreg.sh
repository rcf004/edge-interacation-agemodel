#!/bin/bash

# ensure venv/uv/conda is active before running or add activation/source here

DataDir=$1
Subj=$2
Regname=$3
BehRating=$4
AddReg=$5


python ets_sub_linreg_prewhiten.py ${DataDir} ${Subj} ${Regname} ${BehRating} ${AddReg}

