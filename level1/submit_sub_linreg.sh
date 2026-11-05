#!/bin/bash

# shell script that submits python regression script in parallel to SLURM scheduler
accountID=ACCNUM 		# required for IU slurm submission
email=email@org.com 	# for SLURM email notifications

# specify directory where raw data (parcellated timeseries) are located 
datadir="/path/to/data"

# specify behavior name- should align with input behavior of interest. ie. awk_hrf.npy
beh="awk"

curTime=`date +"%Y%m%d-%H%M%S"`     # pull date for -o/-e logs
mkdir -p ./logs/${curTime}_${beh}_edge_regression_slurmlogs_step   # create logs directory

filelist=../../subject_lists/sublist.tsv

subjid=($(awk -F '\t' '{print $1}' ${filelist} | awk 'NR!=1 {print}'))

BehRating="individual"  #individual to use individual regressors, consensus to use consensus regressor

AddReg=""
# options for AddReg currently:
# "" 								    - left blank: no additional regressor inclusion
# "/path/to/HRF_convolved_regressor"    - add additional control covariate regressor

for i in "${subjid[@]}"; do
    sbatch \
            -A ${accountID} \
            --mail-user=${email} \
            --mail-type=END,FAIL,ARRAY_TASKS \
            --nodes=1 \
            --mem=32G \
            -o ./logs/${curTime}_${beh}_edge_regression_slurmlogs_step/output_${i}.txt \
            -e ./logs/${curTime}_${beh}_edge_regression_slurmlogs_step/error_${i}.txt \
            -J ${i} \
            --time=1:00:00 \
            ./sublevel_linreg.sh \
            ${datadir} ${i} ${beh} ${BehRating} ${AddReg}
            sleep 2

done

