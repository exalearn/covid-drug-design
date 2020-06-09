"""Utility functions for high-throughput evaluation"""
from multiprocessing import Pool
from typing import Iterator, List, NoReturn
from csv import DictReader, DictWriter
import gzip
import sys
import os

from rdkit import Chem
from rdkit.Chem import QED, Crippen
from tqdm import tqdm
import tensorflow as tf
import numpy as np


from molgym.utils.conversions import convert_smiles_to_nx
from molgym.mpnn.data import combine_graphs, convert_nx_to_dict

# Load in SA_scorer
sa_score = '/home/wardlt/miniconda3/envs/covid_dqn/share/RDKit/Contrib/SA_Score/sascorer.py'
if not os.path.isfile(sa_score):
    raise ValueError('SA_scorer file not found. You must edit `baseline.py` to point to the right place')
sys.path.append(os.path.dirname(sa_score))
from sascorer import calculateScore


def load_molecules(path: str, chunk_size: int = 1024) -> Iterator[List[dict]]:
    """Load in a chunk of molecules
    
    Args:
        path (str): Path to the search space
        chunk_size (int): Number of molecules to load
    """
    
    with open(path) as fp:
        reader = DictReader(fp, fieldnames=['source', 'identifier', 'smiles'])
        
        # Loop through chunks
        chunk = []
        for entry in reader:
            chunk.append(entry)
            
            # Return chunk if it is big enough
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []

        # Yield what remains
        yield chunk


def compute_rdkit(gen: Iterator[List[dict]]) -> Iterator[List[dict]]:
    """Compute properties from RDKit for each molecule in a chunk"""
    with Pool() as p:
        for chunk in gen:
            chunk = p.map(_rdkit_eval, chunk, chunksize=256)
            yield chunk


def _rdkit_eval(entry: dict) -> dict:
    """Computes the chemical properties from RDKit,
    adds them to the input dictionary"""
    mol = Chem.MolFromSmiles(entry['smiles'])
    entry['logP'] = Crippen.MolLogP(mol)
    entry['QED'] = QED.qed(mol)
    entry['SA_score'] = calculateScore(mol)
    return entry


def compute_ic50(gen: Iterator[List[dict]], model: tf.keras.Model, atom_types: List[int],
                 bond_types: List[str]) -> Iterator[List[dict]]:
    """Compute the IC50 of a chunk of molecules"""

    for chunk in gen:
        # Get the features for each molecule
        batch = []
        tested_mols = []
        for i, entry in enumerate(chunk):
            graph = convert_smiles_to_nx(entry['smiles'])
            try:
                graph_dict = convert_nx_to_dict(graph, atom_types, bond_types)
            except AssertionError:
                continue
            batch.append(graph_dict)
            tested_mols.append(i)

        # Prepare in input format
        keys = batch[0].keys()
        batch_dict = {}
        for k in keys:
            batch_dict[k] = np.concatenate([np.atleast_1d(b[k]) for b in batch], axis=0)
        inputs = combine_graphs(batch_dict)

        # Compute the IC50
        ic50 = model.predict_on_batch(inputs).numpy()[:, 0]

        # Store in in the chunk data
        for i, v in zip(tested_mols, ic50):
            chunk[i]['pIC50_mpnn'] = v

        yield chunk


def _flat_map(gen: Iterator[List[dict]]) -> dict:
    """Only really used to make the update timer more sensical"""
    for chunk in gen:
        for e in chunk:
            yield e


def write_output(gen: Iterator[List[dict]], path: str) -> NoReturn:
    """Write the output of a processing pipeline to disk"""

    # Get the first entry
    gen = _flat_map(gen)
    entry = next(gen)

    with gzip.open(path, 'wt') as fp:
        # Write the header and first entry
        writer = DictWriter(fp, entry.keys())
        writer.writeheader()
        writer.writerow(entry)

        # Keep writing rows
        for entry in tqdm(gen):
            writer.writerow(entry)
