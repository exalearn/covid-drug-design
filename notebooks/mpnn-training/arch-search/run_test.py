from argparse import ArgumentParser
import json
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks as cb
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import numpy as np

from molgym.mpnn.callbacks import LRLogger, EpochTimeLogger
from molgym.mpnn.layers import GraphNetwork, Squeeze
from molgym.mpnn.data import make_data_loader


def build_fn(atom_features: int = 64, message_steps: int = 8, readout_fn: str = 'mean',
             atomic_contributions: bool = True):
    node_graph_indices = Input(shape=(1,), name='node_graph_indices', dtype='int32')
    atom_types = Input(shape=(1,), name='atom', dtype='int32')
    bond_types = Input(shape=(1,), name='bond', dtype='int32')
    connectivity = Input(shape=(2,), name='connectivity', dtype='int32')

    # Squeeze the node graph and connectivity matrices
    snode_graph_indices = Squeeze(axis=1)(node_graph_indices)
    satom_types = Squeeze(axis=1)(atom_types)
    sbond_types = Squeeze(axis=1)(bond_types)

    output = GraphNetwork(atom_type_count, bond_type_count, atom_features, message_steps,
                          output_layer_sizes=[512, 256, 128],
                          attention_mlp_sizes=[256, 128] if readout_fn == 'attention' else None,
                          atomic_contribution=atomic_contributions, reduce_function=readout_fn,
                          name='mpnn')([satom_types, sbond_types, snode_graph_indices, connectivity])

    # Scale the output
    output = Dense(1, activation='linear', name='scale')(output)

    return Model(inputs=[node_graph_indices, atom_types, bond_types, connectivity],
                 outputs=output)


if __name__ == "__main__":
    # Define the command line arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--readout-function', help='Method used to reduce atomic to molecular features',
                            type=str, choices=['attention', 'softmax', 'max', 'mean', 'sum'], default='sum')
    arg_parser.add_argument('--atomic', help='Use an atomic contribution model', action='store_true')

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__

    # Determine the output directory
    test_dir = os.path.join('results', f'{args.readout_function}_{"atomic" if args.atomic else "molecular"}')
    os.makedirs(test_dir)
    with open(os.path.join(test_dir, 'config.json'), 'w') as fp:
        json.dump(run_params, fp)

    # Making the data loaders
    train_loader = make_data_loader(os.path.join('..', 'train_data.proto'), shuffle_buffer=1024)
    test_loader = make_data_loader(os.path.join('..', 'test_data.proto'))
    val_loader = make_data_loader(os.path.join('..', 'val_data.proto'))

    # Load in the bond and atom type information
    with open('../atom_types.json') as fp:
        atom_type_count = len(json.load(fp))
    with open('../bond_types.json') as fp:
        bond_type_count = len(json.load(fp))

    # Make the model
    model = build_fn(atom_features=256, message_steps=8,
                     readout_fn=args.readout_function, atomic_contributions=args.atomic)

    # Set the scale for the output parameter
    ic50s = np.concatenate([x[1].numpy() for x in iter(train_loader)], axis=0)
    model.get_layer('scale').set_weights([np.array([[ic50s.std()]]), np.array([ic50s.mean()])])

    # Train the model
    final_learn_rate = 1e-6
    init_learn_rate = 1e-3
    decay_rate = (final_learn_rate / init_learn_rate) ** (1. / (1024 - 1))

    def lr_schedule(epoch, lr):
        return lr * decay_rate
    model.compile(Adam(init_learn_rate), 'mean_squared_error', metrics=['mean_absolute_error'])
    history = model.fit(
        train_loader, validation_data=val_loader, epochs=1024, verbose=True,
        shuffle=False, callbacks=[
            LRLogger(),
            EpochTimeLogger(),
            cb.LearningRateScheduler(lr_schedule),
            cb.ModelCheckpoint(os.path.join(test_dir, 'best_model.h5'), save_best_only=True),
            cb.EarlyStopping(patience=128, restore_best_weights=True),
            cb.CSVLogger(os.path.join(test_dir, 'train_log.csv')),
            cb.TerminateOnNaN()
        ]
    )

    # Run on the validation set and assess statistics
    y_true = np.hstack([x[1].numpy()[:, 0] for x in iter(test_loader)])
    y_pred = np.squeeze(model.predict(test_loader))

    pd.DataFrame({'true': y_true, 'pred': y_pred}).to_csv(os.path.join(test_dir, 'val_results.csv'), index=False)

    with open(os.path.join(test_dir, 'val_summary.json'), 'w') as fp:
        json.dump({
            'r2_score': float(np.corrcoef(y_true, y_pred)[1, 0] ** 2),  # float() converts from np.float32
            'spearmanr': float(spearmanr(y_true, y_pred)[0]),
            'kendall_tau': float(kendalltau(y_true, y_pred)[0]),
            'mae': float(np.mean(np.abs(y_pred - y_true))),
            'rmse': float(np.sqrt(np.mean(np.square(y_pred - y_true))))
        }, fp, indent=2)
