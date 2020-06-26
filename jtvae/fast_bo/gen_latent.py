import sys
sys.path.append('../')
import torch
import torch.nn as nn
from optparse import OptionParser
from tqdm import tqdm
import rdkit
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles, RDKFingerprint
from rdkit.Chem import rdmolops, QED
import numpy as np  
from fast_jtnn import *
from fast_jtnn import sascorer
import networkx as nx
import os
import os.path as op
import pandas as pd

### IC50 MPNN imports
import json
from tensorflow.keras.models import load_model
from molgym.envs.rewards.mpnn import MPNNReward
from molgym.mpnn.layers import custom_objects
from molgym.utils.conversions import convert_rdkit_to_nx


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
        #print(mols[i])
        m = MolFromSmiles(mols[i])
        G = convert_rdkit_to_nx(m)    
        reward = MPNNReward(model, atom_types=atom_types, bond_types=bond_types, maximize=False)
        scores.append(reward._call(G))
    return scores

def similarity_search(fps_db, smile):
    fps_test = RDKFingerprint(MolFromSmiles(smile))
    ts=[]
    for i, s_top in enumerate(fps_db):
        ts.append(DataStructs.FingerprintSimilarity(s_top, fps_test))
    ts=np.array(ts)
    return ts.mean() # ts.max()

def scorer(smiles, pIC50_weight, QED_weight, logP_weight, SA_weight, cycle_weight, sim_weight):
    smiles_rdkit = []
    for i in range(len(smiles)):
        smiles_rdkit.append(
            MolToSmiles(MolFromSmiles(smiles[i]), isomericSmiles=True))

    # calculate IC50 of training set using MPNN
    #IC50_scores=calculateScore(smiles_rdkit)

    # read in IC50 of training set from database
    IC50_scores = np.loadtxt('../data/covid/ic50-fulltrain.txt')
    IC50_scores = [x for x in IC50_scores]
    IC50_scores_normalized = (np.array(IC50_scores) - np.mean(IC50_scores)) / np.std(IC50_scores)

    if sim_weight != 0:
        # df_100 = list of molecules to match similarity
        df_100 = pd.read_csv('../data/covid/MPro_6wqf_A_ProteaseData_smiles_top100.csv')
        ms_db = [MolFromSmiles(x) for x in df_100['SMILES'].tolist()]
        fps_db = [RDKFingerprint(x) for x in ms_db]

        sim_values = []
        for i in range(len(smiles)):
            sim_values.append(
                similarity_search(fps_db, smiles_rdkit[i]))
        sim_values_normalized = (
            np.array(sim_values) - np.mean(sim_values)) / np.std(sim_values)
    else:
        sim_values, sim_values_normalized = [], []
        for i in range(len(smiles)):
            sim_values.append(0)
            sim_values_normalized.append(0)
        sim_values_normalized=np.array(sim_values_normalized)
    
    logP_values = []
    for i in range(len(smiles)):
        logP_values.append(
            Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[i])))

    qed_values = []
    for i in range(len(smiles)):
        qed_values.append(
            QED.qed(MolFromSmiles(smiles_rdkit[i])))

    SA_scores = []
    for i in range(len(smiles)):
        SA_scores.append(
            -sascorer.calculateScore(MolFromSmiles(smiles_rdkit[i])))

    cycle_scores = []
    for i in range(len(smiles)):
        cycle_list = nx.cycle_basis(
            nx.Graph(
                rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[i]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_scores.append(-cycle_length)

    SA_scores_normalized = (
        np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
    qed_values_normalized = (
        np.array(qed_values) - np.mean(qed_values)) / np.std(qed_values)
    cycle_scores_normalized = (
        np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)
    logP_values_normalized = (
        np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)

    targets = (pIC50_weight * IC50_scores_normalized + 
               logP_weight * logP_values_normalized +
               SA_weight * SA_scores_normalized +
               QED_weight * qed_values_normalized +
               cycle_weight * cycle_scores_normalized + 
               sim_weight * sim_values_normalized)
   
    return (IC50_scores, qed_values, logP_values, SA_scores, cycle_scores, sim_values, targets)


def main_gen_latent(data_path, vocab_path,
                    model_path, output_path='./',
                    hidden_size=450, latent_size=56,
                    depthT=20, depthG=3, batch_size=100, 
                    pIC50_weight=0, QED_weight=0, logP_weight=1,
                    SA_weight=1, cycle_weight=1, sim_weight=0):
    with open(data_path) as f:
        smiles = f.readlines()
    
    if os.path.isdir(output_path) is False:
        os.makedirs(output_path)

    for i in range(len(smiles)):
        smiles[i] = smiles[i].strip()

    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        latent_points = []
        for i in tqdm(range(0, len(smiles), batch_size)):
            batch = smiles[i:i + batch_size]
            mol_vec = model.encode_from_smiles(batch)
            latent_points.append(mol_vec.data.cpu().numpy())

    latent_points = np.vstack(latent_points)

    IC50_scores, qed_values, logP_values, SA_scores, cycle_scores, sim_values, targets = scorer(smiles, pIC50_weight, QED_weight, logP_weight, SA_weight, cycle_weight, sim_weight)
    np.savetxt(
        os.path.join(output_path, 'latent_features.txt'), latent_points)
    np.savetxt(
        os.path.join(output_path, 'targets.txt'), targets)
    np.savetxt(
        os.path.join(output_path, 'pIC50_values.txt'), np.array(IC50_scores))
    np.savetxt(
        os.path.join(output_path, 'logP_values.txt'), np.array(logP_values))
    np.savetxt(
        os.path.join(output_path, 'QED_values.txt'), np.array(qed_values))
    np.savetxt(
        os.path.join(output_path, 'SA_scores.txt'), np.array(SA_scores))
    np.savetxt(
        os.path.join(output_path, 'cycle_scores.txt'), np.array(cycle_scores))
    np.savetxt(
        os.path.join(output_path, 'sim_values.txt'), np.array(sim_values))
    np.savetxt(
        os.path.join(output_path, 'score_weights.txt'), np.array([pIC50_weight, QED_weight, logP_weight, SA_weight, cycle_weight, sim_weight]))


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-a", "--data", dest="data_path")
    parser.add_option("-v", "--vocab", dest="vocab_path")
    parser.add_option("-m", "--model", dest="model_path")
    parser.add_option("-o", "--output", dest="output_path", default='./')
    parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
    parser.add_option("-l", "--latent", dest="latent_size", default=56)
    parser.add_option("-t", "--depthT", dest="depthT", default=20)
    parser.add_option("-g", "--depthG", dest="depthG", default=3)
    parser.add_option("-q", "--qed", dest="QED_weight", default=0)
    parser.add_option("-x", "--logp", dest="logP_weight", default=1)
    parser.add_option("-y", "--sa", dest="SA_weight", default=1)
    parser.add_option("-z", "--cycle", dest="cycle_weight", default=1)
    parser.add_option("-p", "--pic50", dest="pIC50_weight", default=0)
    parser.add_option("-s", "--sim", dest="sim_weight", default=0)
    opts, args = parser.parse_args()

    hidden_size = int(opts.hidden_size)
    latent_size = int(opts.latent_size)
    depthT = int(opts.depthT)
    depthG = int(opts.depthG)
    QED_weight = float(opts.QED_weight)
    logP_weight = float(opts.logP_weight)
    SA_weight = float(opts.SA_weight)
    cycle_weight = float(opts.cycle_weight)
    pIC50_weight = float(opts.pIC50_weight)
    sim_weight = float(opts.sim_weight)

    main_gen_latent(opts.data_path, opts.vocab_path,
                    opts.model_path, output_path=opts.output_path,
                    hidden_size=hidden_size, latent_size=latent_size,
                    depthT=depthT, depthG=depthG, pIC50_weight=pIC50_weight,
                    QED_weight=QED_weight, logP_weight=logP_weight,
                    SA_weight=SA_weight, cycle_weight=cycle_weight, 
                    sim_weight=sim_weight)
