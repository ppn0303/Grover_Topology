import argparse
import time
import sys
import csv
import pandas as pd
from rdkit import Chem

sys.path.append('./')
from grover.topology.chemutils import *
from grover.topology.mol_tree import *

parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--datapath', type=str, default='./data/zinc/all.txt',
                        help='root directory of dataset. For now, only classification.')
parser.add_argument('--output_clique', type=str, default='./clique.txt',
                        help='filename to output the pre-trained model')
parser.add_argument('--output_data', type=str, default='./data/zinc/all_edit.txt',
                        help='filename to output deleted data')
args = parser.parse_args()

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

data = pd.read_csv(args.datapath)
data_len = len(data)
print(data_len)

num=0
cset = set()
counts = {}

print("start")
s_time = time.time()

for i in range(data_len):
    if num%10000==0:print(f'process : {num} / {data_len}')
    smiles = data.smiles[num]
    try : 
        mol = Chem.MolFromSmiles(smiles)
        mol.GetNumHeavyAtoms()
        moltree = MolTree(smiles)
        for node in moltree.nodes:
            cset.add(node.smiles)
            if node.smiles not in counts:
                counts[node.smiles] = 1
            else:
                counts[node.smiles] += 1
    except : 
        print(f'error smiles is {smiles}')
        data=data.drop(num)
    num += 1


print("Preprocessing Completed!")
t_time = time.time() - s_time
print(f'total time is {t_time:.4f}s, data length is {data_len} -> {len(data)}')

clique_list = list(cset)

data.to_csv(args.output_data, index=False)

with open(args.output_clique, 'w') as file:
    for c in clique_list:
        file.write(c)
        file.write('\n')