import pickle
import argparse
import os.path as op
from os import listdir
import rdkit
from rdkit.Chem import MolFromSmiles, MolToSmiles
import numpy as np
import json
from tensorflow.keras.models import load_model
from molgym.envs.rewards.mpnn import MPNNReward
from molgym.mpnn.layers import custom_objects
from molgym.utils.conversions import convert_rdkit_to_nx

p = argparse.ArgumentParser()
p.add_argument("--path", type=str, required=True, help="Path to results.")
p.add_argument("--mpnn", type=str, default='../../notebooks/mpnn-training/', help="Path to MPNN")
args = p.parse_args()

def calculateScore(mols, mpnn_dir):
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

onlyfiles = [f for f in listdir(args.path) if f.endswith('.pkl')]

# save smiles as txt and get pIC50
smilefiles = [f for f in onlyfiles if "smile" in f]

for filename in smilefiles:
    with open(op.join(args.path,filename), 'rb') as f:
        x = pickle.load(f)
    
    score = calculateScore(x, args.mpnn)
    score = [str(i)+'\n' for i in score]

    x = [i+'\n' for i in x]
    
    with open(op.join(args.path,filename[:-4]+'.txt'), 'w') as f:
        f.writelines(x)

    with open(op.join(args.path,filename[:-4]+'_pIC50.txt'), 'w') as f:
        f.writelines(score)

# save scores as txt
scorefiles = [f for f in onlyfiles if "scores" in f]

for filename in scorefiles:
    with open(op.join(args.path,filename), 'rb') as f:
        x = pickle.load(f)

    x = [str(i)+'\n' for i in x]

    with open(op.join(args.path,filename[:-4]+'.txt'), 'w') as f:
        f.writelines(x)




