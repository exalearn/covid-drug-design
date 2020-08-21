"""Get a list of molecules to use as initial seeds for the reinforcement learning"""

from molgym.utils.conversions import convert_smiles_to_nx, convert_nx_to_smiles
from molgym.mpnn.data import convert_nx_to_dict
from argparse import ArgumentParser
import json
import os

import pandas as pd


# Make the argument parser
parser = ArgumentParser()
parser.add_argument('input_file', help='CSV file with a list of drug-like molecules')
parser.add_argument('--bond-types', help='Path to bond type list',
                    default=os.path.join('notebooks', 'mpnn-training', 'bond_types.json'))
parser.add_argument('--atom-types', help='Path to bond type list',
                    default=os.path.join('notebooks', 'mpnn-training', 'atom_types.json'))
parser.add_argument('--smiles-col', help='Column with the SMILES string', default='InChI')
parser.add_argument('--ranking-col', help='Column with the ranking value', default='pIC50')
parser.add_argument('--descending', help='Sort the molecules in descending rank', action='store_true')
parser.add_argument('--num-mols', nargs='+', help='Number of top molecules to save', default=[100, 1000], type=int)
args = parser.parse_args()

# Load in the file
data = pd.read_csv(args.input_file)
print(f'Loaded {len(data)} molecules')

# Load the bond and atom types
with open(args.atom_types) as fp:
    atom_types = json.load(fp)
with open(args.bond_types) as fp:
    bond_types = json.load(fp)

# Make sure the data parses into ML-ready dict
def parses(s): 
    s = convert_nx_to_smiles(convert_smiles_to_nx(s))
    try:
        convert_nx_to_dict(convert_smiles_to_nx(s), atom_types, bond_types)
        return True
    except ValueError:
        return False
data = data[data[args.smiles_col].apply(parses)]
print(f'Removed molecules that fail to parse. New total: {len(data)}')

# Sort to put the best molecules up top
data.sort_values(args.ranking_col, ascending=not args.descending, inplace=True)
print(f'Sorted all molecules by {args.ranking_col} in {"descending" if args.descending else "ascending"} order')

# Save the top molecules
for top in args.num_mols:
    top_mols = data.head(top)
    os.makedirs('seed-molecules', exist_ok=True)
    with open(os.path.join('seed-molecules', f'top_{top}_{args.ranking_col}.json'), 'w') as fp:
        json.dump(top_mols[args.smiles_col].tolist(), fp)
