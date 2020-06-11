import numpy as np
import json
import os
from argparse import ArgumentParser

from rdkit.Chem import MolFromSmiles
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from molgym.envs.rewards.mpnn import MPNNReward
from molgym.envs.rewards.rdkit import LogP, QEDReward, SAScore, CycleLength
from molgym.mpnn.layers import custom_objects


# Make all of the reward functions
from molgym.utils.conversions import convert_smiles_to_nx

mpnn_dir = os.path.join('notebooks', 'mpnn-training')
model = load_model(os.path.join(mpnn_dir, 'best_model.h5'), custom_objects=custom_objects)
with open(os.path.join(mpnn_dir, 'atom_types.json')) as fp:
    atom_types = json.load(fp)
with open(os.path.join(mpnn_dir, 'bond_types.json')) as fp:
    bond_types = json.load(fp)
rewards = {
    'logP': LogP(),
    'ic50': MPNNReward(model, atom_types, bond_types, maximize=True),
    'QED': QEDReward(),
    'SA': SAScore(),
    'cycles': CycleLength()
}

if __name__ == "__main__":
    # Parse the inputs
    parser = ArgumentParser()
    parser.add_argument("smiles_file")
    args = parser.parse_args()

    # Load in the molecules
    with open(args.smiles_file) as fp:
        mols = [x.strip() for x in fp]

    # Get only the molecules that parse with RDKit
    mols = [x for x in mols if MolFromSmiles(x) is not None]

    # Compute the reward function statistics for all the rewards
    stats = {}
    for name, reward in rewards.items():
        data = [reward(convert_smiles_to_nx(mol)) for mol in tqdm(mols, desc=name)]
        stats[name] = {
            'mean': np.mean(data),
            'scale': np.std(data)
        }

    # Save as a json file
    with open('reward_ranges.json', 'w') as fp:
        json.dump(stats, fp, indent=2)
