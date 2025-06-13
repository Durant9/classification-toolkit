# Classification-toolkit
A modular PyTorch pipeline for training and evaluating image classifiers with cross-validation and hyperparameter optimization.

## Preparation 
Before any launch, prepare this 3 files in your working folder:
:one: class2names.json: a json file that specify, for each numeric class in the dataset, its name:
```
{
  "1": "1st_class_name",
  "2": "2nd_class_name",
  ...
}
```
:two: A pickle file with the name of the dataset that contains a dict. The dict's keys are the numeric classes of the dataset, the values are lists of PIL images with instances of each class:
```
{
  1: [C1_image1, C1_image2, ...],
  2: [C2_image1, C2_image2, ...],
  ...
}
```
:three: A YAML configuration file in a folder called 'config'. Check the folder in the repo for template examples

## Main codes
### Cross Validation
### Train-only
### Fast train&test


## TODO:
- Classification ensemble through bagging
- Simple holdout classification (train-val-test training and validation)
