# Cluster learning-assisted directed evolution (CLADE). 
[![DOI](https://zenodo.org/badge/368001786.svg)](https://zenodo.org/badge/latestdoi/368001786)
**This is the source code of paper: _"Cluster learning-assisted directed evolution" by Yuchi Qiu, Jian Hu, and Guo-Wei Wei. Nature Computational Science 2021."_**

CLADE guides experiments in directed evolution to optimize fitness of variants in a combinatorial library from multi-site mutagenesis. It first utilizes unsupervised clustering to select potential informative variants for experimental screen to obtain their fitness. Then it takes these labeled variants as training data to train a supervised learning model. The supervised learning model predicts fitness of the rest of variants in the combinatorial library. Top-predicted variants potentially have high priority to be screened.

Many MLDE methods can integrate with CLADE. Various sampling strategies can be used for the sampling in selected clusters, including random sampling and Gaussian process. The zero-shot predictions can be used to restrict training data sampling within the top-ranked variants. 

# Table of Contents  

- [Installment](#installment)
- [Usage](#usage)
  * [Encoding](#encoding)
  * [Clustering Sampling](#cluster-learning-sampling)
  * [CLADE](#CLADE)
  * [DEMO](#DEMO)
- [Sources](#sources) 
- [Reference](#reference) 

# Installment
`git clone --recurse-submodules https://github.com/YuchiQiu/CLADE.git`

Then install [MLDE](https://github.com/fhalab/MLDE#building-an-alignment-for-msa-transformer) for supervised learning model:
```
cd CLADE/ 
git clone --recurse-submodules https://github.com/fhalab/MLDE.git`
```
Other packages required:
1. Python3.6 or later.
2. [scikit-learn](https://scikit-learn.org/stable/)
3. numpy
4. pandas
5. pickle

The GB1 dataset needs to be downloaded from: [GB1](https://elifesciences.org/articles/16965)

Put `GB1.xlsx` in the directory `Input/`. 

# Input Files:
`$COMB_LIB.xlsx`: Variants and their fitness in the combinatory library. Only variants with available experimentally determined fitness are listed. First Column (Variants): sequences for variants at mutation sites. Second Column (Fitness): Fitness values.\
`$DATASET_zeroshot.csv`: optional if zero-shot predictions are used. It can be calculated followed by instructions in [ftMLDE](https://github.com/fhalab/MLDE#building-an-alignment-for-msa-transformer). One column of this file name 'Combo" provides the list of all variants. Other columns named by the zero-shot methods give the preditions for each variant. Two zero-shot predictions files used for CLADE paper for GB1 and PhoQ datasets are provided in folder `Input/`. 

# Usage
## Encoding
`Encoding.py` generate sequence encoding tensor for the combinatorial library
```python
$ python3 Encoding.py --help
```
### Inputs:
`--save_dir SAVE_DIR`   Directory for Output Encoding Files. Default value is 'Input/' which store all input files for CLADE  \
`--dataset DATASET`     Name of the data set. Options: 1. GB1; 2. PhoQ. It will load file COMB_LIB.xlsx \
`--encoding ENCODING`  Name of the encoding method; Options: 1. AA; 2.Georgiev. Default: AA \
`--input_dir INPUT_DIR` Directory for Input Files (directory for xlsx files for combinatorial library). Default: Input/\
### Outputs:
`$encoding$.npy` and `$encoding$_normalized.npy` are 3D encoding tensor for combinatorial library. The later one is standardized with zero mean and unit variance. 
`ComboToIndex_$dataset$_$encoding$.pkl`: a dictionary link variant and its index in the xlsx file (`comb_lib`).   
### Examples:
`python3 Encoding.py --encoding AA`

## Clustering Sampling
`clustering_sampling.py` Use hierarchical clustering to generate training data. 
```python
$ python3 clustering_sampling.py --help
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
`--acquisition ACQUISITION` Acquisition function used for in-cluster sampling; default UCB. Options: 1. UCB; 2. epsilon; 3. Thompson; 4. random. Default: random \
`--sampling_para SAMPLING_PARA` Float parameter for the acquisition function. 1. beta for GP-UCB; 2. epsilon for epsilon greedy; 3&4. redundant for Thompson and random sampling. Default: 4.0\
`--use_zeroshot USE_ZEROSHOT` Whether to employ zeroshot predictor in sampling. Default: FALSE \
`--zeroshot ZEROSHOT` Name of zeroshot predictor; Required a CSV file stored in directory $INPUT_PATH with name: $DATA_SET_zeroshot.csv. Default: EvMutation \
`--N_zeroshot N_ZEROSHOT` Number of top ranked variants from zeroshot predictor used for the recombined library. Default: 1600

### Outputs:
`parameters.csv`: Hyperparameters used in the clustering sampling.
`InputValidationData.csv`: Selected labeled variants. Training data for downstream supervised learning. Default will generate 384 labeled variants with batch size 96.
`clustering.npz`: Indecis of variants in each cluster.
### Examples:
`python3 clustering_sampling.py 30 40 40`
## CLADE
`CLADE.py` Run full process of CLADE. Run `clustering_sampling.py` and downstream supervised learning (MLDE).

### Inputs
It requires the same positional and optional arguments with `clustering_sampling.py`. 

It has an additional optional argument:

`--mldepara MLDEPARA`   List of MLDE parameters. Default: MldeParameters.csv 
### Outputs:
In additional to three output files from `clustering_sampling.py`, there are 6 files output from MLDE package. The most important one is: `PredictedFitness.csv` showing predicted fitness of all variants in the combinatorial library. The variants with higher predicted fitness have higher priority to be screened.
### Examples:
`python3 CLADE.py 30 40 40 --batch_size 96 --num_first_round 96 --hierarchy_batch 96 --num_batch 4`
## DEMO:
Functions `Encoding.py` and `clustering_sampling.py` can be run within a few minutes on a desktop. Demo can be run via the examples given above. 

`CLADE.py` includes the ensembled supervised learning models with hyperparameter optimization, which takes a few hours to run on a desktop. A simple demo can be run with a minimized supervised model with only one model without any hyperparameter optimization by using `Demo_MldeParameters.csv`:

`python3 CLADE.py 30 40 40 --batch_size 96 --num_first_round 96 --hierarchy_batch 96 --num_batch 4 --mldepara Demo_MldeParameters.csv`

# Sources
## GB1 dataset
GB1 dataset (`GB1.xlsx`) can be obtained from: [Wu, Nicholas C., et al. "Adaptation in protein fitness landscapes is facilitated by indirect paths." Elife 5 (2016): e16965.](https://elifesciences.org/articles/16965)
## PhoQ dataset
PhoQ dataset (`PhoQ.xlsx`) is owned by Michael T. Laub's lab. Please cite: [Podgornaia, Anna I., and Michael T. Laub. "Pervasive degeneracy and epistasis in a protein-protein interface." Science 347.6222 (2015): 673-677.](https://science.sciencemag.org/content/347/6222/673.abstract)
## MLDE packages and zero-shot predictions
The supervised learning package MLDE and zero-shot predictions can be found in: [Wittmann, Bruce J., Yisong Yue, and Frances H. Arnold. "Informed training set design enables efficient machine learning-assisted directed protein evolution." Cell Systems (2021).](https://www.cell.com/cell-systems/fulltext/S2405-4712(21)00286-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2405471221002866%3Fshowall%3Dtrue)
# Reference
Cluster learning-assisted directed evolution. Yuchi Qiu, Jian Hu, and Guo-Wei Wei. Nature Computational Science 2021.
