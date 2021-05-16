import numpy as np
import os
import sys
import pandas as pd
from sklearn.cluster import KMeans
import warnings
import pickle
import copy



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
def cluster_sample(seed, input_path,save_dir, dataset,encoding,features, AACombo, Fitness, num_training_data,num_first_round,batch_size,hierarchy_batch,N_hierarchy,K_increments):
    np.random.seed(seed)

    # new hierarchy needs to be generated when number of samples is included in the array
    new_hierarchy = np.arange(1,N_hierarchy)*hierarchy_batch+num_first_round

    hierarchy = 0
    n_clusters=K_increments[hierarchy]

    total_clusters = n_clusters
    Index = run_Clustering(features, n_clusters)


    Index = shuffle_index(Index)
    # store sampling results
    Fit_list = []
    SEQ_list = []
    Cluster_list=[]
    #  store ground truth information in each cluster
    Fit = [[] for _ in range(len(Index))]
    SEQ = [[] for _ in range(len(Index))]
    SEQ_index = [[] for _ in range(len(Index))]
    num = 0

    Prob = np.ones([n_clusters]) / n_clusters
    while num < num_first_round:
        cluster_id = np.random.choice(np.arange(0, n_clusters), p=Prob)
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

    while num < num_training_data:
        cluster_id = np.random.choice(np.arange(0, total_clusters), p=Prob)
        # if all samples in one cluster have been selected, we need to update the sampling probablity.
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

        # update sampling probabilities
        if np.mod(num, batch_size) == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                Mean_Fit = np.asarray([np.asarray(Fit[i]).mean() for i in range(total_clusters)])
            Mean_Fit[np.where(np.isnan(Mean_Fit))[0]] = 0
            Prob = Mean_Fit / np.sum(Mean_Fit)
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

    dict_1=pickle.load(open(os.path.join(input_path, 'ComboToIndex'+ '_'+dataset +'_'+ encoding+'.pkl'),'rb'))
    for seq in SEQ_list:
        for cluster_id in range(len(SEQ_index)):
            if dict_1.get(seq) in SEQ_index[cluster_id]:
                Cluster_list.append(cluster_id)

    Cluster_list=np.asarray(Cluster_list)
    sub_data = pd.DataFrame({'AACombo': SEQ_list, 'Fitness': Fit_list,'Cluster': Cluster_list})
    trainingdata=os.path.join(save_dir , 'InputValidationData.csv')
    sub_data.to_csv(trainingdata, index=False)


    np.savez(os.path.join(save_dir, 'clustering.npz'), tree=tree)
    return trainingdata
def main_sampling(seed,args):

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
    save_dir=args.save_dir
    groundtruth_file=os.path.join(input_path, dataset+  '.xlsx')
    groundtruth = pd.read_excel(groundtruth_file)
    Fitness = groundtruth['Fitness'].values
    Fitness = Fitness / Fitness.max()
    AACombo = groundtruth['Variants'].values


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    para=['# training data','# first round','batch size','hierarchy batch','max hierarchy']
    value=[num_training_data,num_first_round,batch_size,hierarchy_batch,N_hierarchy]
    for i in range(len(K_increments)):
        para.append('K'+str(i+1))
        value.append(K_increments[i])
    para_csv = pd.DataFrame({'Parameters': para, 'Value':value })

    para_csv.to_csv(save_dir + 'parameters.csv', index=False)

    # get feature matrix
    encoding_lib = os.path.join(input_path, dataset+'_'+encoding + '_normalized.npy')
    features = np.load(encoding_lib)
    if len(features.shape) == 3:
        features = np.reshape(features, [features.shape[0], features.shape[1] * features.shape[2]])
    features = features[0:len(Fitness)]


    trainingdata=cluster_sample(seed, input_path,save_dir, dataset,encoding,features, AACombo, Fitness, num_training_data,num_first_round,batch_size,hierarchy_batch,N_hierarchy,K_increments)
    return trainingdata,save_dir
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

    args = parser.parse_args()


    # random seed for reproduction
    seed=100
    main_sampling(seed,args)

