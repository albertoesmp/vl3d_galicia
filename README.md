# VirtuaLearn3D Galicia (VL3D-GAL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Welcome to the fork of the VirtuaLearn3D (VL3D) framework for artificial 
intelligence applied to point-wise tasks in 3D point clouds in the region of
Galicia (Northwest Spain).




## Fork
This version of the 
[VL3D framework](https://github.com/3dgeo-heidelberg/virtualearn3d) 
is a fork made for the sake of
reproducibility. Research on the entire region of Galicia from the
ALS PNOA-II dataset has been done with this version. If you are interested in
using the VL3D framework in your own research or works you are strongly adviced
to get it from the official (and updated) repository:

[VirtuaLearn3D++ official repository](https://gitlab.com/catallactical/vl3dpp)

Note that the updated version in the official repository is expected to present
better runtimes, more models, further methods, and many bug fixes. Moreover,
the **VirtuaLearn3D++ is the supported version** right now while **VirtuaLearn3D is
deprecated** and it is available for traceability purposes only.




## Extra files
In this section we comment the JSON files and scripts included in this fork.

### JSON files 

The JSON files are contained inside *cesga/galicia*. The last ones that correspond
with the final experiments in the paper are those at *cesga/galicia/fast_parallel*.
The SLURM scripts used to launch these JSON files at CESGA FinisTerrae-III are also
included here.

### Scripts

```bash
scripts/
```

Contains many scripts used to handle the data from the experiments and the
automatic generation of SQL scripts to insert the results in the database.


```bash
scripts/adv_eval
```

Contains the script to compute the uncertainty assessment from the summarized
results (which are containd inside the *data* subdirectory).


```bash
ddbb/scripts
```

Contains the util *sql_launcher.py* that can be used to run SQL scripts from
a bash terminal.


```bash
ddbb/scripts/init
```

Contains the SQL scripts used to initialize the database.

```bash
ddbb/scripts/query
```

Contains the SQL scripts used to query the database.


### Model and database files

[Zenodo with the models and database files.](https://doi.org/10.5281/zenodo.17956435)


## Cite
**DOI**:  WAITING FOR PUBLICATION

**Bibtex**:
```
WAITING FOR PUBLICATION
```







# Original documentation

Below we reproduce the install and usage instructions from the README in the
original repository.

## Install

### Machine learning install

Since the VL3D framework is a Python-based software, the installation is quite
simple. It consists of three steps (four if you are using conda/miniconda):


1. Clone the repository.

    ```bash
    git clone https://github.com/albertoesmp/vl3d_galicia
    ```

2. Change the working directory to the framework's folder.

    ```bash
    cd vl3d_galicia
    ```

#### Using pip

3. Install the requirements.

    ```bash
    pip install -r requirements.txt
    ```

#### Using conda or miniconda

3. Install the requirements.

    1. In Windows

        ```bash
        conda env create -f vl3d_win.yml
        ```

    2. In Linux.

        ```bash
        conda env create -f vl3d_lin.yml
        ```

4. Activate the environment

   ```bash
   conda activate vl3d
   ```


### Deep learning install

If you are interested in deep learning, you will need some extra steps
to be able to use the GPU with TensorFlow+Keras. You can see the
[TensorFlow documentation](https://www.tensorflow.org/install/pip)
on how to meet the hardware requirements when installing with
pip. Below you can find a summary of the steps that work in the general case:

4. Check that your graphic card supports CUDA. You can check it
    [here](https://developer.nvidia.com/cuda-gpus).

5. Install the drivers of your graphic card.

6. Install CUDA. See either the
[Linux documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/contents.html)
or the
[Windows documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
on how to install CUDA.

7. Install cuDNN. See the
[cuDNN installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).





## Usage


### Pipelines

In the VL3D framework, you can define pipelines with many components that
handle the computation of the different steps involved in a typical artificial
intelligence workflow. These components can be used to compute point-wise
features, train models, predict with trained models, evaluate the predictions,
write the results, apply data transformations, define imputation strategies,
and automatic hyperparameter tuning.

For example, a pipeline could start computing some
geometric features for a point-wise characterization, then train a model,
evaluate its performance with k-folding, and export it as a predictive
pipeline that can later be used to classify other point clouds. Pipelines
are specified in a JSON file, and they can be executed with the command below:


```bash
python vl3d.py --pipeline pipeline_spec.json
```





### Deep learning test

If you are interested in the deep learning components of the VL3D framework,
check that your TensorFlow+Keras installation is correctly linked
to the GPU. Otherwise, deep learning will be infeasible. To check this,
run the following command:



```bash
python vl3d.py --test
```

Note that if the **Keras, Tensorflow, GPU test** does not pass then the framework
will work on the CPU as the GPU is not accessible.

