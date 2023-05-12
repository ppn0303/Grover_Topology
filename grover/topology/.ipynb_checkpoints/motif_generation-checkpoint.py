import torch
import torch.nn as nn
from grover.topology.mol_tree import Motif_Vocab, MolTree
# add this directly in below     from nnutils import create_var
from grover.topology.dfs import Motif_Generation_dfs
from grover.topology.bfs import Motif_Generation_bfs
from grover.topology.chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols, atom_equal, decode_stereo
import rdkit
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import copy, math

from torch.autograd import Variable

def create_var(tensor, requires_grad=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   #"cuda:" + "1" 다중 처리 때문에 이렇게 했나봐 ㅡㅡ 원본 : torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if requires_grad is None:
        return Variable(tensor).to(device)
    else:
        return Variable(tensor, requires_grad=requires_grad).to(device)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


class Motif_Generation(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depth, device, order):
        super(Motif_Generation, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.device = device
        if order == 'dfs':
            self.decoder = Motif_Generation_dfs(vocab, hidden_size, self.device)
        elif order == 'bfs':
            self.decoder = Motif_Generation_bfs(vocab, hidden_size, self.device)

    def forward(self, mol_batch, node_rep):
        set_batch_nodeID(mol_batch, self.vocab)

        word_loss, topo_loss, word_acc, topo_acc = self.decoder(mol_batch, node_rep)

        #loss = word_loss + topo_loss

        return word_loss, topo_loss, word_acc, topo_acc

