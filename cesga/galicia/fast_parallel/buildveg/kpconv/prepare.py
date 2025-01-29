import json
import sys
import os

# Extract args
jsonf_path = sys.argv[1]

# Generate preparation JSON
with open(jsonf_path, 'r') as jsonf:  # READ Json file
    jsond = json.load(jsonf)  # Dictionary from JSON file
    seqpipe = jsond['sequential_pipeline']
    # Find train and predictive pipeline writer components in sequence
    train, writer = None, None
    for comp in seqpipe:
        if 'train' in comp:
            train = comp
        if 'out_pipeline' in comp:
            writer = comp
    # Update pretrained_model spec
    outdir = jsond['out_pcloud'][0]
    outdir = outdir.replace('*', '')
    outpipe = writer['out_pipeline'].replace('*', '')
    oldpipe = os.path.join(outdir, outpipe)
    oldkeras = oldpipe.replace('.pipe', '.keras')
    oldmodel = oldpipe.replace('.pipe', '.model')
    train['pretrained_model'] = oldmodel
    train['pretrained_nn_path'] = oldkeras
    train['model_args']['pre_processing']['support_strategy_fast'] = 4
    train['model_args']['pre_processing']['receptive_field_oversampling'] = {
        "min_points": 2,
        "strategy": "nearest",
        "k": 3,
        "radius": 2.5
    }
    train['model_args']['model_handling']['training_epochs'] = 0
    # Update predictive pipeline wirter
    newdir = outpipe.replace('pipe/', 'prep_pipe/')
    writer['out_pipeline'] = f'*{newdir}'
    # Export preparation JSON
    prep_path = jsonf_path.replace('training', 'prepare')
    with open(prep_path, 'w') as prepf:
        json.dump(jsond, prepf)
    print(f'Preparation JSON exported to: "{prep_path}"')
