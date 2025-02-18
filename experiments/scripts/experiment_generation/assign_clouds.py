import os
import laspy
import numpy as np
import json
import glob

#################
### Constants ###
#################
VL3D_DIR = os.path.abspath('/mnt/netapp2/Store_uni/home/usc/ci/myg/Codigos/vl3d_galicia_dev')
CLOUDS_PATH = os.path.join(os.environ.get('PNOA'), 'mined')
EXPERIMENTS_PATH = os.path.join(VL3D_DIR, 'experiments')
CLOUDS_ASSIGMENT_JSON= os.path.join(EXPERIMENTS_PATH, 'clouds_assigment.json')

CLOUDS_PER_JOB=4

# Number of point clouds per person
n_miguel = 96
n_silvia = 96
n_samuel = 95

# create path if it does not exist
os.makedirs(EXPERIMENTS_PATH, exist_ok=True)

EXPERIMENTS = ["vegetation", "building", "lmhveg", "buildveg"]

DISCARDED_CLOUDS = {
   "vegetation": ['276', '147', '235', '16'],
   "building": ['260', '68', '216', '21'],
   "lmhveg": ['234', '97', '239', '189'],
   "buildveg": ['260', '68', '216', '21'],
}


# Get all experiment files
files = glob.glob(os.path.join(CLOUDS_PATH, 'RGBIr_MERGE_*_minmaxnorm.laz'))

# Assign point clouds for each experiments
for exp in EXPERIMENTS:
    # Failed or used in testing point clouds not to be included in the experiments
    EXPERIMENT_DISCARDED_CLOUDS = [os.path.join(CLOUDS_PATH, f'RGBIr_MERGE_{num}_minmaxnorm.laz') for num in DISCARDED_CLOUDS[exp]]


    # Remove files per experiment
    experiment_files = [f for f in files if f not in EXPERIMENT_DISCARDED_CLOUDS]


    miguel_clouds = experiment_files[0:n_miguel]
    silvia_clouds = experiment_files[n_miguel:n_miguel+n_silvia]
    samuel_clouds = experiment_files[n_miguel+n_silvia:]

    clouds = [miguel_clouds, silvia_clouds, samuel_clouds]

    miguel_clouds_grouped = [miguel_clouds[i:i+CLOUDS_PER_JOB] for i in range(0, len(miguel_clouds), CLOUDS_PER_JOB)]
    silvia_clouds_grouped = [silvia_clouds[i:i+CLOUDS_PER_JOB] for i in range(0, len(silvia_clouds), CLOUDS_PER_JOB)]
    samuel_clouds_grouped = [samuel_clouds[i:i+CLOUDS_PER_JOB] for i in range(0, len(samuel_clouds), CLOUDS_PER_JOB)]

    json_data = {
    'usccimyg': {},
    'usccisra': {},
    'usccisss': {},
    }

    for i, c in enumerate(miguel_clouds_grouped):
        json_data['usccimyg'][str(i)] = c

    for i, c in enumerate(silvia_clouds_grouped):
        json_data['usccisra'][str(i)] = c

    for i, c in enumerate(samuel_clouds_grouped):
        json_data['usccisss'][str(i)] = c
    
    CLOUDS_ASSIGMENT_JSON= os.path.join(EXPERIMENTS_PATH, f'{exp}_clouds_assigment.json')

    with open (CLOUDS_ASSIGMENT_JSON, 'w') as f:
        json.dump(json_data, f, indent=4)