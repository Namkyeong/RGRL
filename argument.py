import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikics", help="Name of the dataset. Supported names are: cora, citeseer, pubmed, cs, computers, photo, and physics")
    parser.add_argument("--embedder", type=str, default="RGRL")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--eval_freq", type=float, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--es", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=10000) 
    parser.add_argument("--dropout", type=float, default=0.0)
    
    parser.add_argument("--aug_params", "-p", nargs="+", default=[
                        0.3, 0.4, 0.3, 0.2], help="Hyperparameters for augmentation (p_f1, p_f2, p_e1, p_e2). Default is [0.3, 0.4, 0.3, 0.2]")
    parser.add_argument("--layers", nargs='+', default= [512, 256], help="The number of units of each layer of the GNN. Default is [512, 256]")
    parser.add_argument("--pred_hid", type=int, default=512, help="The number of hidden units of layer of the predictor. Default is 512")
    
    ## Number of samples
    parser.add_argument("--sample", type=int, default=256, help="The number of global sample. Default is 256")
    parser.add_argument("--topk", type=int, default=4, help="The number of local sample. Default is 4")

    ## Temperature Hyperparameters
    parser.add_argument("--temp_t", type=float, default=0.01, help="Global temperature for target network")
    parser.add_argument("--temp_s", type=float, default=0.1, help="Global temperature for online Network")
    parser.add_argument("--temp_t_diff", type=float, default=1.0, help="Local temperature for target network")
    parser.add_argument("--temp_s_diff", type=float, default=0.1, help="Local temperature for online network")

    ## Hyperparameters for inverse degree sampling distribution
    parser.add_argument("--alpha", type=float, default=0.9, help="Hyperparameters for the skewness of inverse degree sampling distribution")
    parser.add_argument("--beta", type=float, default=0.0, help="Hyperparameters for the minimum of inverse degree sampling distribution")
    
    ## Hyperparameters for loss function
    parser.add_argument("--lam", type=float, default=1.0, help="controls the weight between local and global")

    return parser.parse_known_args()


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        ## Hyperparameters
        if name not in ['root', 'device', 'eval_freq', 'es', 'epochs', 'dropout', 'layers', 'pred_hid', 'mad']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)