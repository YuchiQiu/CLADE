from cluster_learning_sampling import main_sampling
import os

def mlde(args,save_dir,trainingdata):
    input_path = args.input_path
    encoding = args.encoding
    dataset = args.dataset
    encoding_lib = os.path.join(input_path, dataset+'_'+encoding+'_normalized.npy')
    combo_to_index =os.path.join(input_path,'ComboToIndex' + '_'+dataset +'_'+ encoding+ '.pkl')
    mldepara=os.path.join(input_path,args.mldepara)
    os.system('python3 ExecuteMlde.py ' +trainingdata +' '+ \
              encoding_lib +' '+combo_to_index+' --model_params '+mldepara +' --output '+save_dir+' --hyperopt')
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


    ## parameters for MLDE
    parser.add_argument("--mldepara",help="List of MLDE parameters",default='MldeParameters.csv')

    args = parser.parse_args()


    # random seed for reproduction
    seed=100
    trainingdata,save_dir=main_sampling(seed,args)

    mlde(args, save_dir,trainingdata)






