import sys
sys.path.append('../')

import rdkit
from rdkit.Chem import MolFromSmiles, MolToSmiles
import numpy as np  
import os
import os.path as op

import json
from tensorflow.keras.models import load_model
from molgym.envs.rewards.mpnn import MPNNReward
from molgym.mpnn.layers import custom_objects
from molgym.utils.conversions import convert_rdkit_to_nx

import argparse

def calculateScore(mols, mpnn_dir='../../notebooks/mpnn-training/'):
    scores=[]
    # load model and atom/bond types
    model = load_model(op.join(mpnn_dir, 'model.h5'), custom_objects=custom_objects)
    with open(op.join(mpnn_dir, 'atom_types.json')) as fp:
        atom_types = json.load(fp)
    with open(op.join(mpnn_dir, 'bond_types.json')) as fp:
        bond_types = json.load(fp)
    # calculate score for each mol
    for i in range(len(mols)):
        m = MolFromSmiles(mols[i])
        G = convert_rdkit_to_nx(m)    
        reward = MPNNReward(model, atom_types=atom_types, bond_types=bond_types, maximize=False)
        scores.append(reward._call(G))
    return scores



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, required=True, help="File with molecules.")
    args = p.parse_args()
    
    with open(op.join(args.file), 'r') as f:
        x = f.readlines()
    
    score = calculateScore(x)
    score = [str(i)+'\n' for i in score]
    
    with open(op.join(args.file[:-4]+'_pIC50.txt'), 'w') as f:
        f.writelines(score)
