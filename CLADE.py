from clustering_sampling import main_sampling
import os
import pandas as pd
def mlde(args,save_dir,trainingdata):
    input_path = args.input_path
    encoding = args.encoding
    dataset = args.dataset
    encoding_lib = os.path.join(input_path, dataset+'_'+encoding+'_normalized.npy')
    combo_to_index =os.path.join(input_path,'ComboToIndex' + '_'+dataset +'_'+ encoding+ '.pkl')
    mldepara=os.path.join(input_path,args.mldepara)

    os.system('python3 MLDE/execute_mlde.py ' +trainingdata +' '+ \
              encoding_lib +' '+combo_to_index+' --model_params '+mldepara +' --output '+save_dir +' --hyperopt')
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
    parser.add_argument('--acquisition',help="Acquisition function used for in-cluster sampling; default UCB. Options: 1. UCB; 2. epsilon; 3. Thompson; 4. random. ",default='random')
    parser.add_argument('--sampling_para', help="Float parameter for the acquisition function. 1. beta for GP-UCB; 2. epsilon for epsilon greedy; 3&4. redundant for Thompson and random sampling",type=float, default= 4.0)

    parser.add_argument('--use_zeroshot',help="Whether to employ zeroshot predictor in sampling. Default: FALSE",type=bool, default=False)
    parser.add_argument('--zeroshot',help="name of zeroshot predictor; Required a CSV file stored in directory $INPUT_PATH with name: $DATA_SET_zeroshot.csv. Default: EvMutation",default='EvMutation')
    parser.add_argument('--N_zeroshot',help="Number of top ranked variants from zeroshot predictor used for the recombined library. Default: 1600",type=int,default=1600)

    ## parameters for MLDE
    parser.add_argument("--mldepara",help="List of MLDE parameters; Default: MldeParameters.csv",default='MldeParameters.csv')

    args = parser.parse_args()


    # random seed for reproduction
    seed=args.seed

    trainingdata=main_sampling(seed,args,args.save_dir)
    mlde(args, args.save_dir,trainingdata)






