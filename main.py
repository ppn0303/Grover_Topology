import random, os

import numpy as np
import torch
from rdkit import RDLogger
from socket import gethostname

from grover.util.parsing import parse_args, get_newest_train_args
from grover.util.utils import create_logger
from task.cross_validate import cross_validate, randomsearch, gridsearch, make_confusion_matrix
from task.fingerprint import generate_fingerprints, generate_embvec
from task.predict import make_predictions, write_prediction
from task.pretrain import pretrain_model, subset_learning
from grover.data.torchvocab import MolVocab

from grover.topology.mol_tree import *

#add for gridsearch
from argparse import ArgumentParser, Namespace

import torch.distributed as dist

def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
class process_tracker(object):
    def __init__(self, args):
        self.args = args
        self.num_subset = 0
        self.now_subset = 0
        self.now_iter = 0
        self.origin_data_path = args.data_path
        
        
    def save_process(self):
        path = os.path.join(self.args.save_dir, "process.txt")
        txt = open(path, 'w')
        txt.write("num_subset:%d\n" % (self.num_subset))
        txt.write("now_subset:%d\n" % (self.now_subset))
        txt.write("now_iter:%d\n" % (self.now_iter))
        txt.close()
        print('process saved')
        
    def load_process(self):
        '''
        if you don't have saved data, you must make txt file like below
        
        num_subset:0
        now_subset:0
        now_iter:0
        '''
        path = os.path.join(self.args.save_dir, "process.txt")
        f = open(path, 'r')
        lines = f.readlines()
        self.num_subset = np.int(lines[0].split(':')[1].split('\n')[0])
        self.now_subset = np.int(lines[1].split(':')[1].split('\n')[0])
        self.now_iter = np.int(lines[2].split(':')[1].split('\n')[0])
        f.close()
        

if __name__ == '__main__':
    # setup random seed
    setup(seed=42)

    a = MolVocab

    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Initialize MolVocab
    mol_vocab = MolVocab

    args = parse_args()

    if args.parser_name == 'finetune':
        logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)
        if args.randomsearch==True :  randomsearch(args, logger)
        elif args.gridsearch==True : gridsearch(args, logger)
        else : cross_validate(args, logger)
    elif args.parser_name == 'pretrain':
        logger = create_logger(name='pretrain', save_dir=args.save_dir)
        if args.subset_learning==True : 
            pt = process_tracker(args)
            pt.load_process()
            subset_learning(args, logger, pt)
        else : 
            pretrain_model(args, logger)
    elif args.parser_name == "eval":
        logger = create_logger(name='eval', save_dir=args.save_dir, quiet=False)
        if args.confusionmatrix==True : make_confusion_matrix(args, logger)
        else : cross_validate(args, logger)
    elif args.parser_name == 'fingerprint':
        train_args = get_newest_train_args()
        logger = create_logger(name='fingerprint', save_dir=None, quiet=False)
        if args.embvec==True : 
            atom_vec, bond_vec = generate_embvec(args, logger)
            if args.fingerprint_source=='atom':
                feas = atom_vec
            else : 
                feas = bond_vec
        else : 
            feas = generate_fingerprints(args, logger)
        
        np.savez_compressed(args.output_path, fps=feas)
    elif args.parser_name == 'predict':
        train_args = get_newest_train_args()
        avg_preds, test_smiles = make_predictions(args, train_args)
        write_prediction(avg_preds, test_smiles, args)       
        
