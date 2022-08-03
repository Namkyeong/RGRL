import torch
torch.set_num_threads(2)

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)

import numpy as np
np.random.seed(0)

import argument

def main():
    args, unknown = argument.parse_args()
    
    if args.embedder == 'RGRL':
        from models import RGRL_ModelTrainer
        embedder = RGRL_ModelTrainer(args)

    embedder.train()
    embedder.writer.close()

if __name__ == "__main__":
    main()


