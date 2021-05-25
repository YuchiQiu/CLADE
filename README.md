# CLADE: Cluster learning-assisted directed evolution. 
CLADE guides experiments in directed evolution to optimize fitness of variants in a combinatorial library from multi-site mutagenesis. It first utilizes unsupervised clustering to select potential informative variants for experimental screen to obtain their fitness. Then it takes these labeled variants as training data to train a supervised learning model. The supervised learning model predicts fitness of the rest of variants in the combinatorial library. Top-predicted variants potentially have high priority to be screened.

# Table of Contents  

- [Requirements](#requirements)
- [Usage](#usage)
  * [Encoding](#encoding)
  * [Cluster-learning Sampling](#cluster-learning-sampling)
  * [CLADE](#CLADE)
- [Sources](#sources) 
- [Reference](#reference) 


# Requirements
1. Python3.6 or later.
2. [scikit-learn](https://scikit-learn.org/stable/)
3. numpy
4. pandas
5. pickle
6. [MLDE](https://github.com/fhalab/MLDE) (all required packages for supervised learning are given in MLDE)


# Usage
## Encoding
`Encoding.py` generate sequence encoding tensor for the combinatorial library
```python
$ python3 Encoding.py --help
```
### Inputs:
`--save_dir SAVE_DIR`   Directory for Output Encoding Files. Default value is 'Input/' which store all input files for CLADE  \
`--comb_lib COMB_LIB`   xlsx file for unlabeled combinatorial library. \
`--dataset DATASET`     Name of the data set. Options: 1. GB1; 2. PhoQ. It will load file COMB_LIB_all.xlsx \
`--encoding ENCODING`  Name of the encoding method; Options: 1. AA; 2.Georgiev. Default: AA \
`--input_dir INPUT_DIR` Directory for Input Files (directory for xlsx files for combinatorial library). \
### Outputs:
`$encoding$.npy` and `$encoding$_normalized.npy` are 3D encoding tensor for combinatorial library. The later one is standardized with zero mean and unit variance. 
`ComboToIndex_$dataset$_$encoding$.pkl`: a dictionary link variant and its index in the xlsx file (`comb_lib`).   
### Examples:
`python3 AAencoding.py --encoding AA`

## Cluster-learning Sampling
`cluster_learning_sampling.py` Use hierarchical clustering to generate training data. 
```python
$ python3 cluster_learning_sampling.py --help
```
### Inputs
#### positional arguments:  
`K_increments` Increments of clusters at each hierarchy; Input a list; For example: --K_increments 30 30 30. \
#### optional arguments: 
`--dataset DATASET`     Name of the data set. Options: 1. GB1; 2. PhoQ. \
`--encoding ENCODING`  Name of the encoding method; Options: 1. AA; 2.Georgiev. Default: AA \
`--num_first_round NUM_FIRST_ROUND` number of variants in the first round sampling; Default: 96  \
`--batch_size BATCH_SIZE` Batch size. Number of variants can be screened in parallel. Default: 96  \
`--hierarchy_batch HIERARCHY_BATCH` Excluding the first-round sampling, new hierarchy is generated after every hierarchy_batch variants are collected until max hierarchy. Default:96  \
`--num_batch NUM_BATCH` number of batches; Default: 4  \
`--input_path INPUT_PATH`  Input Files Directory. Default 'Input/'  \
`--save_dir SAVE_DIR`   Output Files Directory; Default: current time  \

### Outputs:
`parameters.csv`: Hyperparameters used in the cluster-learning sampling.
`InputValidationData.csv`: Selected labeled variants. Training data for downstream supervised learning. Default will generate 384 labeled variants with batch size 96.
`clustering.npz`: Indecis of variants in each cluster.
### Examples:
`python3 cluster_learning_sampling.py 30 40 40`
## CLADE
`CLADE.py` CLADE framework. Run `cluster_learning_sampling.py` and downstream supervised learning (MLDE).
```python
$ python3 CLADE.py --help
```
### Inputs
#### positional arguments:  
`K_increments` Increments of clusters at each hierarchy; Input a list; For example: --K_increments 30 30 30. \
#### optional arguments: 
`--dataset DATASET`     Name of the data set. Options: 1. GB1; 2. PhoQ. \
`--encoding ENCODING`  Name of the encoding method; Options: 1. AA; 2.Georgiev. Default: AA \
`--num_first_round NUM_FIRST_ROUND` number of variants in the first round sampling; Default: 96  \
`--batch_size BATCH_SIZE` Batch size. Number of variants can be screened in parallel. Default: 96  \
`--hierarchy_batch HIERARCHY_BATCH` Excluding the first-round sampling, new hierarchy is generated after every hierarchy_batch variants are collected until max hierarchy. Default:96  \
`--num_batch NUM_BATCH` number of batches; Default: 4  \
`--input_path INPUT_PATH`  Input Files Directory. Default 'Input/'  \
`--save_dir SAVE_DIR`   Output Files Directory; Default: current time  \
`--mldepara MLDEPARA`   List of MLDE parameters \
### Outputs:
Additional to `cluster_learning_sampling.py`, output files contain 6 additional files from MLDE outputs. The most important one is: `PredictedFitness.csv` showing predicted fitness of all variants in the combinatorial library. The variants with higher predicted fitness have higher priority to be screened.
### Examples:
`python3 cluster_learning_sampling.py 30 40 40 --batch_size 96 --num_first_round 96 --hierarchy_batch 96 --num_batch 4`
# Sources
## GB1 dataset
GB1 dataset (`GB1.xlsx`) can be downloaded from: [Wu, Nicholas C., et al. "Adaptation in protein fitness landscapes is facilitated by indirect paths." Elife 5 (2016): e16965.](https://elifesciences.org/articles/16965)
## PhoQ dataset
PhoQ dataset (`PhoQ.xlsx`) is owned by Michael T. Laub's lab. Refer to [Podgornaia, Anna I., and Michael T. Laub. "Pervasive degeneracy and epistasis in a protein-protein interface." Science 347.6222 (2015): 673-677.](https://science.sciencemag.org/content/347/6222/673.abstract)
## MLDE 
The supervised learning package MLDE: [Wittmann, Bruce J., Yisong Yue, and Frances H. Arnold. "Machine Learning-Assisted Directed Evolution Navigates a Combinatorial Epistatic Fitness Landscape with Minimal Screening Burden." (2020).](https://www.biorxiv.org/content/10.1101/2020.12.04.408955v1)
# Reference
This work is under peer review.
