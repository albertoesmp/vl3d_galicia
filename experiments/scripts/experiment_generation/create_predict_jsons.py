import json
import os
import copy

########################
### GLOBAL CONSTANTS ###
########################
VL3D_DIR = os.path.abspath('/mnt/netapp2/Store_uni/home/usc/ci/myg/Codigos/vl3d_galicia_dev')

class ExperimentParameters:
    def __init__(self, name, model_name, model_id, model_dir, json_template):
        self.name = name
        self.model_name = model_name
        self.model_id = model_id
        self.model_dir = model_dir
        self.json_template = json_template


vegetation_exp = ExperimentParameters("vegetation", 
                                      "sflnet_try1_XIrRGB_vegetation_T5", 
                                      21, 
                                      "/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_vegetation/T5/prep_pipe",
                                      os.path.join(VL3D_DIR, "experiments", "scripts", "experiment_generation", "sflnet_try1_XIrRGB_vegetation_prepared_predict_T5.json"))

building_exp = ExperimentParameters("building", 
                                    "sflnet_try1_XIrRGB_building_T5", 
                                    41, 
                                    "/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_building/T5/prep_pipe",
                                    os.path.join(VL3D_DIR, "experiments", "scripts", "experiment_generation", "sflnet_try1_XIrRGB_building_prepared_predict_T5.json"))

lmhveg_exp = ExperimentParameters("lmhveg", 
                                  "sflnet_try1_XIrRGB_lmhveg_T5", 
                                  61, 
                                  "/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_lmhveg/T5/prep_pipe",
                                  os.path.join(VL3D_DIR, "experiments", "scripts", "experiment_generation", "sflnet_try1_XIrRGB_lmhveg_prepared_predict_T5.json"))

buildveg_exp = ExperimentParameters("buildveg", 
                                    "sflnet_try1_XIrRGB_buildveg_T5", 
                                    81, 
                                    "/mnt/netapp2/Store_uscciaep/lidar_data/pnoa2/vl3d/sflnet_try1_XIrRGB_buildveg/T5/prep_pipe",
                                    os.path.join(VL3D_DIR, "experiments", "scripts", "experiment_generation", "sflnet_try1_XIrRGB_buildveg_prepared_predict_T5.json"))

########################
### GLOBAL CONSTANTS ###
########################
OUT_PATH='path_to_output_clouds'
PIPE_PATH='path_to_model_pipe'
NN_PATH='path_to_model_nn'

experiment_parameters = [vegetation_exp, building_exp, lmhveg_exp, buildveg_exp]
for exp_par in experiment_parameters:

    PREDICT_JSON_PATH = exp_par.json_template
    EXPERIMENT_NAME = exp_par.name
    CLOUDS_ASSIGMENT_PATH = os.path.join(VL3D_DIR, 'experiments', f'{exp_par.name}_clouds_assigment.json')


    # load predict_template.json
    with open(PREDICT_JSON_PATH, 'r') as f:
        predict_template = json.load(f)

    # load clouds_assigment.json
    with open(CLOUDS_ASSIGMENT_PATH, 'r') as f:
        clouds_assigment = json.load(f)

    for user in clouds_assigment:
        output_json_dir = os.path.join(VL3D_DIR, 'experiments', 'spec', exp_par.name, user)
        os.makedirs(output_json_dir, exist_ok=True)

        for idx in clouds_assigment[user]:
            currJson = copy.deepcopy(predict_template)
            currJson['in_pcloud'] = clouds_assigment[user][idx]
            out_pcloud = []

            for cloud in clouds_assigment[user][idx]:
                cloud_name = cloud.split("/")[-1]
                parts = cloud_name.split("_")
                cloud_name = "_".join(parts[1:3])
                out_pcloud.append(os.path.join(OUT_PATH, cloud_name, "*"))
            
            currJson['out_pcloud'] = out_pcloud
            currJson['sequential_pipeline'][1]['model_path'] = PIPE_PATH

            with open(os.path.join(output_json_dir, f'{user}_execution_{idx}.json'), 'w') as f:
                json.dump(currJson, f, indent=4)