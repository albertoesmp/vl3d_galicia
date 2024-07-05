import json
import os
import copy

########################
### GLOBAL CONSTANTS ###
########################
VL3D_DIR = os.path.abspath(os.getenv('VL3D_DIR'))

PREDICT_JSON_PATH = os.path.join(VL3D_DIR, 'scripts', 'experiment_generation', 'predict_template_buildveg.json')
CLOUDS_ASSIGMENT_PATH = os.path.join(VL3D_DIR, 'experiments', 'clouds_assigment.json')

EXPERIMENT_NAME='buildveg'

OUT_PATH='path_to_output_clouds'
PIPE_PATH='path_to_model_pipe'
NN_PATH='path_to_model_nn'

# load predict_template.json
with open(PREDICT_JSON_PATH, 'r') as f:
    predict_template = json.load(f)

# load clouds_assigment.json
with open(CLOUDS_ASSIGMENT_PATH, 'r') as f:
    clouds_assigment = json.load(f)


# Print summary
print("Miguel")
print(f"#Points to process: {clouds_assigment['usccimyg']['nPoints']}")
print(f"#Executions: {len(clouds_assigment['usccimyg']['groups'])}")

nClouds = 0
for cloud in clouds_assigment['usccimyg']['groups'].values():
    nClouds += len(cloud['cloud'])
print(f"#Clouds: {nClouds}")

print("\nSilvia")
print(f"#Points to process: {clouds_assigment['usccisra']['nPoints']}")
print(f"#Executions: {len(clouds_assigment['usccisra']['groups'])}")

nClouds = 0
for cloud in clouds_assigment['usccisra']['groups'].values():
    nClouds += len(cloud['cloud'])
print(f"#Clouds: {nClouds}")

print("\nSamuel")
print(f"#Points to process: {clouds_assigment['usccisss']['nPoints']}")
print(f"#Executions: {len(clouds_assigment['usccisss']['groups'])}")

nClouds = 0
for cloud in clouds_assigment['usccisss']['groups'].values():
    nClouds += len(cloud['cloud'])
print(f"#Clouds: {nClouds}")

# Create execution jsons
for person in clouds_assigment:
    output_json_dir = os.path.join(VL3D_DIR, 'experiments', EXPERIMENT_NAME, person)

    # create directory for each person
    os.makedirs(output_json_dir, exist_ok=True)
    

    for i, group in enumerate(clouds_assigment[person]['groups'].values()):
        currJson = copy.deepcopy(predict_template)
        currJson['in_pcloud'] = group['cloud']
        out_pcloud = []
        for cloud in group['cloud']:
            cloud_name = cloud.split("/")[-1]
            parts = cloud_name.split("_")
            cloud_name = "_".join(parts[1:3])
            out_pcloud.append(os.path.join(OUT_PATH, cloud_name, "*"))

        currJson['out_pcloud'] = out_pcloud
        currJson['sequential_pipeline'][1]['model_path'] = PIPE_PATH
        currJson['sequential_pipeline'][1]['nn_path'] = NN_PATH

        with open(os.path.join(output_json_dir, f'{person}_execution_{i}.json'), 'w') as f:
            json.dump(currJson, f, indent=4)