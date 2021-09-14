# Bicycle Project Evaluation v1.3

## 1. About
This repo functions as a means of an evaluation medium for a given task.

## 2. Setup
### Cloning the repo and supplying data
```
git clone https://github.com/Schmaexxi/qm-bicycle-project-evaluation-v1.3.git
```

Navigate into the cloned repo and create a directory labeled 'data', which should include the specified files for this task


### Dependencies
Make sure to run Python3.9 to be able to replicate this setup effortlessly. Feel free to choose whichever manager you like to install the packages.
#### Pip
```
python -m venv bicycle_project_eval
source bicycle_project_eval/bin/activate
pip install -r requirements.txt 
```

#### Conda
```
conda env create -n bicycle_project_eval -f environment.yml
conda activate bicycle_project_eval
```


## 3. File structure
There exist two scripts:

One for testing at ./tests/test_data_consistency.py

The other for the specified evaluation at ./annotator_and_reference_set_analysis.py

