# bayes_gain_screens
A reduced set of code from previous repositories that focuses purely on producing gain screens for DDF pipeline. 

# Preamble

This repository provides a code base that performs Bayesian infernce of doubly differential total electron content (DDTEC) from Jones scalars, Gaussian process regression DDTEC, and a pipeline that stiches all the pieces together. It is actively maintained and in rolling development. It does not adhere to stable release candidate protocol, so pieces of code may drastically change.

It assumes one has access to the LoTSS-DR2 archive, however as the pipeline evolves this assumption may disappear.

# How it works

This pipeline is a set of steps each of which is a self-contained script that can be called from the command line. The user specifies which steps they would like to execute and the pipeline compiles this into a task execution graph which gets executed via Dijkstraa DFS path. The pipeline maintains the working directories so that the user never needs to clear caches etc. Importantly, the execution environment for each step is controlled using containerisation with singularity. You can by-pass the containerisation by calling the pipeline manually.

# Requirements

You'll want DDF-pipeline resources for everything except the imaging step which requires > 260GB RAM.

1. singularity should be installed on all nodes you may execute on.

2. a singularity image with killMS and DDF: branches used by SKSP (ask Frits Sweijen). Call this sksp_image

3. a singularity image with killMS and DDF: the gainscreens_premerge DDF branch (ask Frits Sweijen). Call this gainscreens_image. Until the SKSP pipeline is updated to use master DDF branch, the gain screens functionality is not available on the same image as sksp_iamge. (See https://github.com/cyriltasse/DDFacet/issues/651)

4. One of the following:

Either, a singularity image of bayes_gain_screens, or build your own with (requires sudo):

```
sudo singularity build bayes_gain_screens.simg bayes_gain_screens.singularity
```

Or, a conda environment will the bayes_gain_screens (call it `tf_py` because it has tensorflow):

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p /opt/miniconda3
. /opt/miniconda3/etc/profile.d/conda.sh
hash -r 
conda config --set auto_activate_base false --set always_yes yes
conda update -q conda
conda info -a

conda create -q -n tf_py python=3.6
conda activate tf_py

cd $HOME
mkdir git
cd git 
git clone https://github.com/Joshuaalbert/bayes_gain_screens.git
cd bayes_gain_screens
pip install -r requirements.txt
pip install --no-deps pyregion pyparsing
python setup.py install
```

5. bash shell


