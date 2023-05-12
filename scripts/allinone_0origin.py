import pickle
import os
import csv
import shutil
import pandas as pd
import time
#import torch
from collections import Counter
from typing import Callable, Union

from argparse import ArgumentParser, Namespace
import numpy as np
from multiprocessing import Pool
from typing import List, Tuple
from tqdm import tqdm
from rdkit import RDLogger

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import grover.util.utils as fea_utils
from grover.util.utils import get_data, makedirs, load_features, save_features
from grover.data.molfeaturegenerator import get_available_features_generators, \
    get_features_generator
from grover.data.task_labels import rdkit_functional_group_label_features_generator
from grover.topology.mol_tree import *

from rdkit import Chem
from descriptastorus.descriptors import rdDescriptors

from grover.data.molfeaturegenerator import register_features_generator

parser = ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help='Path to data CSV')
parser.add_argument("--sample_per_file", type=int, default=1000)
parser.add_argument("--workers", type=int, default=10)
parser.add_argument("--output_path", default="../drug_data/grover_data/delaneyfreesolvlipo")
parser.add_argument('--newstart', action='store_true', default=False,
                    help='Whether to not load partially complete featurization and instead start from scratch')
parser.add_argument('--continue', action='store_true', default=True)
parser.add_argument('--max_data_size', type=int,
                    help='Maximum number of data points to load')
parser.add_argument('--sequential', action='store_true', default=False,
                    help='Whether to task sequentially rather than in parallel')

args = parser.parse_args()

def load_smiles(data_path):
    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        res = []
        for line in reader:
            res.append(line)
    return res, header

def save_smiles(data_path, index, data, header='smiles'):
    fn = os.path.join(data_path, str(index) + ".csv")
    with open(fn, "w") as f:
        fw = csv.writer(f)
        fw.writerow([header])
        for d in data:
            fw.writerow([d])
    f.close()
    
def save_features(data_path, index, features):
    fn = os.path.join(data_path, str(index) + ".npz")
    np.savez_compressed(fn, features=features)
    
def save_moltrees(data_path, index, moltrees):
    fn = os.path.join(data_path, str(index) + ".p")
    with open(fn, 'wb') as file: 
        pickle.dump(moltrees, file)
    file.close()

def save_cliques(data_path, index, cliques):
    clique_path = data_path+f'/clique{index}.txt'
    with open(clique_path, 'w') as file:
        for c in cliques:
            file.write(c)
            file.write('\n')
    file.close()
    
def load_checkpoint(process_path):
    with open(process_path, 'r') as file:
        line1 = file.readline()
        line2 = file.readline()
        temp_i = int(line1.split(' \n')[0])
        temp_num = int(line2)
    file.close()
    return temp_i, temp_num

def load_cliques(data_path, index):
    cliques = set()
    clique_path = args.output_path+f'/cliques/clique{index}.txt'
    with open(clique_path, 'r') as file:
        for line in file:
            cliques.add(line)
    file.close()
    return cliques

Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]

# The functional group descriptors in RDkit.
RDKIT_PROPS = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
               'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
               'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
               'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
               'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
               'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
               'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
               'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
               'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
               'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
               'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
               'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
               'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
               'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
               'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
               'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
               'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

BOND_FEATURES = ['BondType', 'Stereo', 'BondDir']

@register_features_generator('allinone')
def make_fgfeatures_moltree_clique(mol):
    """
    Generates functional group label for a molecule using RDKit.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D features.
    """
    try : 
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        #test GetNumHeavyAtoms()
        mol = Chem.MolFromSmiles(smiles)
        mol.GetNumHeavyAtoms()

        #make fg features
        generator = rdDescriptors.RDKit2D(RDKIT_PROPS)
        features = generator.process(smiles)[1:]
        features = np.array(features)
        features[features != 0] = 1

        #make cliuqe
        mol_tree = MolTree(smiles)
        cset=[]
        for node in mol_tree.nodes:
            cset.append(node.smiles)

        #make moltree
        mol_tree.recover()
        mol_tree.assemble()
    
        return smiles, features, cset, mol_tree
    except : 
        return None, None, None, None
    
#execute

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

debug, info = lg.debug, lg.info

#원본 smiles식 불러오기 및 섞기
data = pd.read_csv(args.data_path)
res = data.smiles

process_path = os.path.join(args.output_path, "process.txt")
graph_path = os.path.join(args.output_path, "graph")
fea_path = os.path.join(args.output_path, "feature")
trees_path = os.path.join(args.output_path, "moltrees")
cliques_path = os.path.join(args.output_path, "cliques")

try : 
    temp_i, num = load_checkpoint(process_path)
    print(f'load_checkpoint is {temp_i}th smiles, {num}th file')
    cliques = load_cliques(cliques_path, num)
    
    temp_i += 1
    num += 1
    res = res[temp_i:]
except : 
    print(f'dont load_checkpoint')
    temp_i = 0
    num = 0
    cliques = set()

n_graphs = len(res)
if n_graphs < args.sample_per_file:
    num -= 1

#경로 생성
if args.newstart:
    os.remove(args.output_path)

os.makedirs(args.output_path, exist_ok=True)
    
#스마일즈를 몇개의 fold로 나눌건지 
nfold = int(n_graphs / args.sample_per_file + 1)
if num>0:
    print("Number of files: %d" % (nfold+num-1))
else : 
    print("Number of files: %d" % (nfold+num))
    
os.makedirs(graph_path, exist_ok=True)
os.makedirs(fea_path, exist_ok=True)
os.makedirs(trees_path, exist_ok=True)
os.makedirs(cliques_path, exist_ok=True)

smiles_list = []
smiles_full_list = []
features_list = []
moltree_list = []

features_generator = get_features_generator('allinone')

if args.sequential:
    mapping = map(features_generator, res)
else:
    mapping = Pool(args.workers).imap(features_generator, res)

count = 0

info("start")
s_time = time.time()

for i, output in tqdm(enumerate(mapping), total=n_graphs):
    if output[0] != None : 
        smiles_list.append(output[0])
        smiles_full_list.append(output[0])
        features_list.append(output[1])
        for j in range(len(output[2])):
            cliques.add(output[2][j])
        moltree_list.append(output[3])
        count+=1
    else : 
        info(f'error smiles is {res[i]}')
        #res=res.drop(i)
    
    if count==args.sample_per_file:
        save_smiles(graph_path, num, smiles_list)
        save_features(fea_path, num, features_list)
        save_moltrees(trees_path, num, moltree_list)
        save_cliques(cliques_path, num, cliques)
        smiles_list = []
        features_list = []
        moltree_list = []
           
        temp_file = open(process_path, 'w')
        temp_file.write(f"{i} \n")
        temp_file.write(f"{num}")
        temp_file.close()
        
        count=0
        num+=1
        

save_smiles(graph_path, num, smiles_list)
save_features(fea_path, num, features_list)
save_moltrees(trees_path, num, moltree_list)
save_cliques(cliques_path, num, cliques)
        
info("Preprocessing Completed!")
t_time = time.time() - s_time
info(f'total time is {t_time:.4f}s, data length is {n_graphs} -> {len(res)}')

#res.to_csv(args.output_path+'/smiles.csv', header='smiles', index=False)
        
clique_list = list(cliques)

save_cliques(args.output_path, '', cliques)

summary_path = os.path.join(args.output_path, "summary.txt")
summary_fout = open(summary_path, 'w')
summary_fout.write("n_files:%d\n" % (nfold+num))
summary_fout.write("n_samples:%d\n" % (n_graphs+temp_i))
summary_fout.write("sample_per_file:%d\n" % args.sample_per_file)
summary_fout.close()

#delete whole files
#save_smiles(graph_path, '_error', errorsmiles_list)
save_smiles(graph_path, '_full', smiles_full_list)
#save_features(fea_path, '_full', features_full_list)
#save_moltrees(trees_path, '_full', moltree_full_list)