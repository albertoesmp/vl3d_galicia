import os
import laspy
import numpy as np
import json

#################
### Constants ###
#################
VL3D_DIR = os.path.abspath(os.getenv('VL3D_DIR'))
CLOUDS_PATH = os.path.join(os.environ.get('PNOA'), 'mined')
EXPERIMENTS_PATH = os.path.join(VL3D_DIR, 'experiments')
CLOUDS_ASSIGMENT_JSON= os.path.join(EXPERIMENTS_PATH, 'clouds_assigment.json')

# create path if it does not exist
os.makedirs(EXPERIMENTS_PATH, exist_ok=True)


MAX_POINTS_PER_GROUP = 500_000_000

total_num_points = 0

# print all files in clouds_path
for root, dirs, files in os.walk(CLOUDS_PATH):
    for file in files:
        with laspy.open(os.path.join(root, file)) as las:
            total_num_points += las.header.point_count

print('Total number of points: ', total_num_points)
points_per_person = total_num_points / 3
print('Points per person: ', points_per_person)


# TODO: Use dictionaries instead of lists
miguel_clouds = {'nPoints': 0, 'nGroups': 0, 'groups': {}}
silvia_clouds = {'nPoints': 0, 'nGroups': 0, 'groups': {}}
samuel_clouds = {'nPoints': 0, 'nGroups': 0, 'groups': {}}

clouds = [miguel_clouds, silvia_clouds, samuel_clouds]
indices = [0, 0, 0]

for root, dirs, files in os.walk(CLOUDS_PATH):
    for file in files:
        cloud_path = os.path.join(root, file)
        with laspy.open(cloud_path) as las:
            num_points = las.header.point_count

            # Get min among the three lists
            min_idx = np.argmin(np.array([cloud['nPoints'] for cloud in clouds]))


            cloud = clouds[min_idx]
            idx = indices[min_idx]

            cloud['nPoints'] += num_points

            if idx not in cloud['groups']:
                cloud['groups'][idx] = {'nPoints': 0, 'cloud': []}

            if cloud['groups'][idx]['nPoints'] + num_points > MAX_POINTS_PER_GROUP:
                idx += 1
                indices[min_idx] = idx
                cloud['groups'][idx] = {'nPoints': 0, 'cloud': []}

            cloud['groups'][idx]['nPoints'] += num_points
            cloud['groups'][idx]['cloud'].append(cloud_path)

clouds[0]['nGroups'] = indices[0] + 1
clouds[1]['nGroups'] = indices[1] + 1
clouds[2]['nGroups'] = indices[2] + 1

json_data = {
    'usccimyg': {
        'nPoints': clouds[0]['nPoints'], 'nGroups': clouds[0]['nGroups'],'groups': clouds[0]['groups']},
    'usccisra': {
        'nPoints': clouds[1]['nPoints'], 'nGroups': clouds[1]['nGroups'],'groups': clouds[1]['groups']},
    'usccisss': {
        'nPoints': clouds[2]['nPoints'], 'nGroups': clouds[2]['nGroups'],'groups': clouds[2]['groups']}
}

with open (CLOUDS_ASSIGMENT_JSON, 'w') as f:
    json.dump(json_data, f, indent=4)