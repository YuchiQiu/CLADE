import numpy as np
import os
import sys
import pandas as pd
from sklearn.cluster import KMeans
import warnings
import pickle
import copy
from sklearn.gaussian_process import GaussianProcessRegressor



def sampling_subcluster_priority(seed,acquisition,sampling_para,features,Fitness,SEQ_index,Index):
    if acquisition in ['UCB', 'epsilon','Thompson']:
        X_GP=[]
        Y_GP=[]
        for cluster_id in range(len(Index)):
            X_GP.extend(features[SEQ_index[cluster_id]])
            Y_GP.extend(Fitness[SEQ_index[cluster_id]])
        X_GP=np.asarray(X_GP)
        Y_GP=np.asarray(Y_GP)
        # print(X_GP.shape)
        # print(Y_GP.shape)
        regr = GaussianProcessRegressor(random_state=seed)
        regr.fit(X_GP, Y_GP)
    for cluster_id in range(len(Index)):
        if len(SEQ_index[cluster_id])>0 and len(Index[cluster_id]):
            if acquisition in ['UCB', 'epsilon','Thompson']:
                # X_GP = features[SEQ_index[cluster_id]]
                # Y_GP = Fitness[SEQ_index[cluster_id]]
                # regr = GaussianProcessRegressor(random_state=seed)
                # regr.fit(X_GP, Y_GP)
                pred_mean, pred_std = regr.predict(features[Index[cluster_id]], return_std=True)
                if acquisition == 'UCB':
                    # beta = 4 followed by Romero, Philip et al., PNAS 2013
                    index_GP = np.argsort(pred_mean + pred_std * np.sqrt(sampling_para))[::-1]
                elif acquisition == 'epsilon':
                    p_GP = np.random.rand()
                    if p_GP < sampling_para:
                        # exploration
                        index_GP = np.argsort(pred_std)[::-1]
                    else:
                        # exploitation
                        index_GP = np.argsort(pred_mean)[::-1]
                elif acquisition =='Thompson':
                    samples_TS = np.random.normal(0,1,pred_mean.shape)
                    samples_TS = pred_mean + pred_std * samples_TS
                    index_GP = np.argsort(samples_TS)[::-1]
                Index[cluster_id]=Index[cluster_id][index_GP]
            elif acquisition == 'random':
                np.random.shuffle(Index[cluster_id])
    return Index
def shuffle_index(Index):
    for i in range(len(Index)):
        np.random.shuffle(Index[i])

    return Index

def run_Clustering( features, n_clusters, subclustering_index=np.zeros([0])):
    if len(subclustering_index) > 0:
        features_sub = features[subclustering_index, :]
    else:
        features_sub=features



    kmeans = KMeans(n_clusters=n_clusters).fit(features_sub)
    cluster_labels = kmeans.labels_


    Length = []
    Index = []

    if len(subclustering_index) > 0:
        for i in range(cluster_labels.max() + 1):
            index = subclustering_index[np.where(cluster_labels == i)[0]]
            l = len(index)
            Index.append(index)
            Length.append(l)
    else:
        for i in range(cluster_labels.max() + 1):
            index = np.where(cluster_labels == i)[0]
            l = len(index)
            Index.append(index)
            Length.append(l)

    return Index


def split_subcluster(features, n_clusters, Index, Fitness, AACombo, SEQ_index, cluster_id):
    subclustering_index = []
    subclustering_index.extend(Index[cluster_id])
    subclustering_index.extend(SEQ_index[cluster_id])
    subclustering_index=np.asarray(subclustering_index)

    Index2 = run_Clustering( features, n_clusters, subclustering_index)
    Fit_sub = [[] for _ in range(n_clusters)]
    SEQ_sub = [[] for _ in range(n_clusters)]
    SEQ_index_sub = [[] for _ in range(n_clusters)]

    Index_sub=copy.deepcopy(Index2)


    for k in SEQ_index[cluster_id]:

        for i in range(len(Index2)):
            if k in Index2[i]:
                Fit_sub[i].append(Fitness[k])
                SEQ_sub[i].append(AACombo[k])
                SEQ_index_sub[i].append(k)
                Index_sub[i]=np.delete(Index_sub[i],np.where(Index_sub[i]==k))

    return Fit_sub, SEQ_sub, SEQ_index_sub, Index_sub


def sample_min_cluster(min_num_cluster, Fitness, AACombo, Index, Fit, SEQ, SEQ_index):

    num_add = 0
    for i in range(len(Index)):
        if len(Fit[i]) < min_num_cluster:
            for k in range(min_num_cluster - len(Fit[i])):
                Fit[i].append(Fitness[Index[i][k]])
                SEQ[i].append(AACombo[Index[i][k]])
                SEQ_index[i].append(Index[i][k])
                num_add += 1

        Index[i] = np.delete(Index[i], list(range(0, min_num_cluster - len(Fit[i]))))
    return Index, Fit, SEQ, SEQ_index, num_add
def length_index(SEQ_index):
    k=0
    for i in range(len(SEQ_index)):
        k+=len(SEQ_index[i])
    return k
def cluster_sample(args,save_dir,features,AACombo, Fitness,ComboToIndex):
    K_increments = args.K_increments
    for i in range(len(K_increments)):
        K_increments[i]=int(K_increments[i])
    N_hierarchy=len(K_increments)
    encoding = args.encoding
    dataset=args.dataset
    num_first_round=int(args.num_first_round)
    batch_size=int(args.batch_size)
    hierarchy_batch=int(args.hierarchy_batch)
    num_batch=int(args.num_batch)
    num_training_data = batch_size*num_batch
    input_path=args.input_path
    seed=args.seed

    acquisition=args.acquisition
    sampling_para=args.sampling_para

    # new hierarchy needs to be generated when number of samples is included in the array
    new_hierarchy = np.arange(1,N_hierarchy)*hierarchy_batch+num_first_round

    hierarchy = 0
    n_clusters=K_increments[hierarchy]

    total_clusters = n_clusters
    Index = run_Clustering(features, n_clusters)


    Index = shuffle_index(Index)
    # store selected samples with sequential order
    Fit_list = []
    SEQ_list = []
    Cluster_list=[]
    #  store selected samples according to the cluster they belong to
    Fit = [[] for _ in range(len(Index))]
    SEQ = [[] for _ in range(len(Index))]
    SEQ_index = [[] for _ in range(len(Index))]
    num = 0

    Prob = np.ones([n_clusters]) / n_clusters
    while num < num_first_round:
        cluster_id = np.random.choice(np.arange(0, total_clusters), p=Prob)
        while len(Index[cluster_id]) == 0:
            Prob[cluster_id] = 0
            Prob = Prob / np.sum(Prob)
            cluster_id = np.random.choice(np.arange(0, total_clusters), p=Prob)
        Fit[cluster_id].append(Fitness[Index[cluster_id][0]])
        SEQ[cluster_id].append(AACombo[Index[cluster_id][0]])
        Fit_list.append(Fitness[Index[cluster_id][0]])
        SEQ_list.append(AACombo[Index[cluster_id][0]])
        SEQ_index[cluster_id].append(Index[cluster_id][0])
        Index[cluster_id] = np.delete(Index[cluster_id], [0])
        num += 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Mean_Fit = np.asarray([np.asarray(Fit[i]).mean() for i in range(len(Index))])
    Mean_Fit[np.where(np.isnan(Mean_Fit))[0]] = 0
    Prob = Mean_Fit / np.sum(Mean_Fit)

    ## use tree structure to store hirearchy
    sample_length = np.zeros([len(Index)])
    for cluster_id in range(len(Index)):
        sample_length[cluster_id] = len(SEQ[cluster_id])
    tree = [[]]

    parents = [-1 * np.ones(len(Index))]
    tree[hierarchy] = {'parents': copy.deepcopy(parents[hierarchy]), 'mean': copy.deepcopy(np.asarray(Mean_Fit)),
                         'num_samples': copy.deepcopy(np.asarray(sample_length)), 'Index': copy.deepcopy(Index),
                         'SEQ_index': copy.deepcopy(SEQ_index)}

    # use GP-UCB or GP-epsilon greedy search or random sampling
    # to get sampling priority for sequences in each non-empty cluster
    Index = sampling_subcluster_priority(seed, acquisition, sampling_para, features, Fitness, SEQ_index, Index)


    while num < num_training_data:
        cluster_id = np.random.choice(np.arange(0, total_clusters), p=Prob)
        # if all sequences in a cluster have been selected,
        # we need to update the sampling probablity by setting the Prob in this cluster to be zero.
        while len(Index[cluster_id]) == 0:
            Prob[cluster_id] = 0
            Prob = Prob / np.sum(Prob)
            cluster_id = np.random.choice(np.arange(0, total_clusters), p=Prob)

        Fit[cluster_id].append(Fitness[Index[cluster_id][0]])
        SEQ[cluster_id].append(AACombo[Index[cluster_id][0]])
        Fit_list.append(Fitness[Index[cluster_id][0]])
        SEQ_list.append(AACombo[Index[cluster_id][0]])
        SEQ_index[cluster_id].append(Index[cluster_id][0])
        Index[cluster_id] = np.delete(Index[cluster_id], [0])

        sample_length[cluster_id] = len(SEQ[cluster_id])

        num += 1

        # update sampling probabilities and update sampling priority
        if np.mod(num, batch_size) == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                Mean_Fit = np.asarray([np.asarray(Fit[i]).mean() for i in range(total_clusters)])
            Mean_Fit[np.where(np.isnan(Mean_Fit))[0]] = 0
            Prob = Mean_Fit / np.sum(Mean_Fit)

            # use GP-UCB or GP-epsilon greedy search or random sampling
            # to get sampling priority for sequences in each non-empty cluster
            Index = sampling_subcluster_priority(seed, acquisition, sampling_para, features, Fitness, SEQ_index, Index)

            tree[hierarchy]['num_samples'] = copy.deepcopy(np.asarray(sample_length))
            tree[hierarchy]['mean'] = copy.deepcopy(np.asarray(Mean_Fit))
            tree[hierarchy]['Index'] = copy.deepcopy(Index)
            tree[hierarchy]['SEQ_index']=copy.deepcopy(SEQ_index)
        # generate new hierarchy
        if num in new_hierarchy:
            hierarchy+=1
            n_clusters_subclustering=K_increments[hierarchy]
            num_new_cluster = np.floor(Prob * n_clusters_subclustering)
            cluster_id = np.where(Mean_Fit == Mean_Fit.max())[0][0]
            num_new_cluster[cluster_id] = num_new_cluster[cluster_id] + n_clusters_subclustering - num_new_cluster.sum()
            parents.append(-1 * np.ones([total_clusters]))
            tree.append([])
            for cluster_id in range(total_clusters):
                if num_new_cluster[cluster_id] >= 1:
                    Fit_sub, SEQ_sub, SEQ_index_sub, Index_sub = \
                        split_subcluster( features, int(num_new_cluster[cluster_id]) + 1, Index, Fitness,
                                         AACombo, SEQ_index, cluster_id)
                    # print(str(cluster_id)+' '+ str(len(Fit_sub)) +' ' +str(int(num_new_cluster[cluster_id])))
                    Fit[cluster_id] = Fit_sub[0]
                    SEQ[cluster_id] = SEQ_sub[0]
                    SEQ_index[cluster_id] = SEQ_index_sub[0]
                    Index[cluster_id] = Index_sub[0]
                    for k in range(1, len(Fit_sub)):
                        Fit.append(Fit_sub[k])
                        SEQ.append(SEQ_sub[k])
                        SEQ_index.append(SEQ_index_sub[k])
                        Index.append(Index_sub[k])

                    parents[hierarchy][cluster_id] = cluster_id
                    parents[hierarchy] = np.append(parents[hierarchy], cluster_id * np.ones([len(Fit_sub) - 1]))

            total_clusters = n_clusters_subclustering + total_clusters

            ## update tree structure and randomly shuffle Index;
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                Mean_Fit = np.asarray([np.asarray(Fit[i]).mean() for i in range(total_clusters)])
            Mean_Fit[np.where(np.isnan(Mean_Fit))[0]] = 0
            Prob = Mean_Fit / np.sum(Mean_Fit)
            Index = shuffle_index(Index)

            sample_length = np.zeros([len(Index)])
            for cluster_id in range(len(Index)):
                sample_length[cluster_id] = len(SEQ[cluster_id])

            tree[hierarchy] = {'parents': copy.deepcopy(parents[hierarchy]), 'mean': copy.deepcopy(np.asarray(Mean_Fit)),
                                 'num_samples': copy.deepcopy(np.asarray(sample_length)), 'Index': copy.deepcopy(Index),
                                 'SEQ_index': copy.deepcopy(SEQ_index)}



    Fit_list = np.asarray(Fit_list)
    SEQ_list = np.asarray(SEQ_list)

    for seq in SEQ_list:
        for cluster_id in range(len(SEQ_index)):
            if ComboToIndex.get(seq) in SEQ_index[cluster_id]:
                Cluster_list.append(cluster_id)


    Cluster_list=np.asarray(Cluster_list)

    sub_data = pd.DataFrame({'AACombo': SEQ_list, 'Fitness': Fit_list,'Cluster': Cluster_list})
    trainingdata=os.path.join(save_dir , 'InputValidationData.csv')
    sub_data.to_csv(trainingdata, index=False)


    np.savez(os.path.join(save_dir, 'clustering.npz'), tree=tree)
    return trainingdata
def main_sampling(seed,args,save_dir):
    np.random.seed(seed)

    if not os.path.exists(save_dir):
        os.system('mkdir -p '+save_dir)
    groundtruth_file=os.path.join(args.input_path, args.dataset+  '.xlsx')
    groundtruth = pd.read_excel(groundtruth_file)
    Fitness = groundtruth['Fitness'].values
    fit_max=Fitness.max()
    # Fitness = Fitness / Fitness.max()
    if args.use_zeroshot:
        AACombo,FIT_zeroshot = library_zeroshot(args.input_path, save_dir, args.dataset, args.zeroshot, args.N_zeroshot)
        from Encoding import RunEncoding

        tmp, features, ComboToIndex = RunEncoding(args.input_path,AACombo,args.encoding)
        print(features.shape)
        Fitness=FIT_zeroshot/fit_max

    else:
        # get feature matrix
        encoding_lib = os.path.join(args.input_path, args.dataset+'_'+args.encoding + '_normalized.npy')
        features = np.load(encoding_lib)
        ComboToIndex=pickle.load(open(os.path.join(args.input_path, 'ComboToIndex'+ '_'+args.dataset +'_'+ args.encoding+'.pkl'),'rb'))
        Fitness = Fitness / Fitness.max()
        AACombo = groundtruth['Variants'].values

    if len(features.shape) == 3:
        features = np.reshape(features, [features.shape[0], features.shape[1] * features.shape[2]])
    features = features[0:len(Fitness)]

    trainingdata=cluster_sample(args,save_dir,features,AACombo, Fitness,ComboToIndex)
    return trainingdata



def library_zeroshot(input_path,save_dir,dataset,zeroshot,N_zeroshot):
    data_landscape = pd.read_excel(os.path.join(input_path,dataset+'.xlsx'))
    SEQ = data_landscape['Variants'].values
    Fitness=data_landscape['Fitness'].values
    # Fitness=Fitness/max(Fitness)

    data_zeroshot = pd.read_csv(os.path.join(input_path,dataset+'_zeroshot.csv'))
    data_zeroshot = data_zeroshot.sort_values(by=zeroshot, ascending=False)

    top_Combo = data_zeroshot['Combo'].values[0:N_zeroshot]

    SEQ_zeroshot = [SEQ[i] for i in range(len(SEQ)) if SEQ[i] in top_Combo]
    FIT_zeroshot = [Fitness[i] for i in range(len(SEQ)) if SEQ[i] in top_Combo]
    return SEQ_zeroshot,FIT_zeroshot
if __name__ == "__main__":

    import argparse

    from time import  strftime
    time = strftime("%Y%m%d-%H%M%S")

    parser = argparse.ArgumentParser()

    parser.add_argument("K_increments", nargs="+", help = "Increments of clusters at each hierarchy; Input a list; For example: --K_increments 30 30 30.")
    parser.add_argument("--dataset", help = "Name of the data set. Options: 1. GB1; 2. PhoQ.", default = 'GB1')
    parser.add_argument("--encoding", help = "encoding method; Option: 1. AA; 2. Georgiev. Default: AA", default = 'AA')
    parser.add_argument("--num_first_round", help = "number of variants in the first round sampling; Default: 96",type=int,default=96)
    parser.add_argument("--batch_size", help = "Batch size. Number of variants can be screened in parallel. Default: 96",type=int,default = 96)
    parser.add_argument("--hierarchy_batch", help = "Excluding the first-round sampling, new hierarchy is generated after every hierarchy_batch variants are collected, until max hierarchy. Default: 96",default = 96)
    parser.add_argument("--num_batch", help="number of batches; Default: 4",type=int,default=4)
    parser.add_argument('--input_path',help="Input Files Directory. Default 'Input/'",default='Input/')
    parser.add_argument('--save_dir', help="Output Files Directory; Default: current time", default= time + '/')
    parser.add_argument('--seed', help="random seed",type=int, default= 100)
    parser.add_argument('--acquisition',help="Acquisition function used for in-cluster sampling; default UCB. Options: 1. UCB; 2. epsilon; 3. Thompson; 4. random. Default: random",default='random')
    parser.add_argument('--sampling_para', help="Float parameter for the acquisition function. 1. beta for GP-UCB; 2. epsilon for epsilon greedy; 3&4. redundant for Thompson and random sampling. Default: 4.0",type=float, default= 4.0)

    parser.add_argument('--use_zeroshot',help="Whether to employ zeroshot predictor in sampling. Default: FALSE",type=bool, default=False)
    parser.add_argument('--zeroshot',help="name of zeroshot predictor; Required a CSV file stored in directory $INPUT_PATH with name: $DATA_SET_zeroshot.csv. Default: EvMutation",default='EvMutation')
    parser.add_argument('--N_zeroshot',help="Number of top ranked variants from zeroshot predictor used for the recombined library. Default: 1600",type=int,default=1600)

    args = parser.parse_args()


    # random seed for reproduction
    seed=args.seed
    main_sampling(seed,args,args.save_dir)

