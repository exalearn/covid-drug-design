import os
import json
import timeit
import logging
import platform
from math import inf
from tqdm import tqdm
from datetime import datetime
from csv import DictWriter
from rdkit import RDLogger
from rdkit.Chem import GetPeriodicTable, MolFromSmiles
from argparse import ArgumentParser
from molgym.agents.moldqn import DQNFinalState
from molgym.agents.preprocessing import MorganFingerprints
from molgym.envs.actions import MoleculeActions
from molgym.envs.rewards import LogP
from molgym.envs.simple import Molecule
from molgym.envs.rewards.mpnn import MPNNReward
from molgym.utils.conversions import convert_nx_to_smiles, convert_smiles_to_nx
from molgym.mpnn.layers import custom_objects
from tensorflow.keras.models import load_model

# Set up the logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.DEBUG)
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)


def get_platform_info():
    """Get information about the computer running this process"""

    return {
        'processor': platform.machine(),
        'python_version': platform.python_version(),
        'python_compiler': platform.python_compiler(),
        'hostname': platform.node(),
        'os': platform.platform(),
        'cpu_name': platform.processor(),
        'n_cores': os.cpu_count()
    }


def run_experiment(episodes, n_steps, update_q_every, log_file):
    """Perform the RL experiment

    Args:
        episodes (int): Number of episodes to run
        n_steps (int): Maximum number of steps per episode
        update_q_every (int): After how many updates to update the Q function
        log_file (DictWriter): Tool to write the output function
    """
    best_reward = -1 * inf

    for e in tqdm(range(episodes), desc='RL Episodes', leave=True, disable=False):
        current_state = env.reset()
        for s in tqdm(range(n_steps), desc='\t RL Steps', disable=True):
            # Get action based on current state
            action = agent.action()

            # Fix cluster action
            new_state, reward, done, _ = env.step(action)

            # Check if it's the last step and flag as done
            if s == n_steps:
                logger.debug('Last step  ... done')
                done = True

            # Save outcome
            agent.remember(current_state, action, reward,
                           new_state, agent.env.action_space.get_possible_actions(), done)

            # Train model
            loss = agent.train()

            # Write to output log
            log_file.writerow({
                'episode': e, 'step': s, 'smiles': convert_nx_to_smiles(env.state),
                'loss': loss, 'reward': reward, 'epsilon': agent.epsilon
            })

            # Update state
            current_state = new_state

            if reward > best_reward:
                best_reward = reward
                logger.info("Best reward: %s" % best_reward)

            if done:
                break

        # Update the Q network after certain numbers of episodes and adjust epsilon
        if e > 0 and e % update_q_every == 0:
            agent.update_target_q_network()
        agent.epsilon_adj()


if __name__ == "__main__":
    # Define the command line arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--epsilon', help='Controls degree of exploration',
                            default=1.0, type=float)
    arg_parser.add_argument('--max-steps', help='Maximum number of steps per episode',
                            default=32, type=int)
    arg_parser.add_argument('--episodes', help='Number of episodes to run',
                            default=200, type=int)
    arg_parser.add_argument('--q-update-freq', help='After how many episodes to update Q network',
                            default=10, type=int)
    arg_parser.add_argument('--reward', help='Which reward function to use.',
                            choices=['ic50', 'logP'], default='ic50')
    arg_parser.add_argument('--hidden-layers', nargs='+', help='Number of units in the hidden layers of the Q network',
                            default=(24, 48, 24), type=int)
    arg_parser.add_argument('--no-backtrack', action='store_true', help='Disallow bond removal')
    arg_parser.add_argument('--initial-molecule', type=str, default=None, help='Starting molecule')

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__

    # Get the list of elements
    #  We want those where SMILES supports implicit valences
    mpnn_dir = os.path.join('notebooks', 'mpnn-training')
    with open(os.path.join(mpnn_dir, 'atom_types.json')) as fp:
        atom_types = json.load(fp)
    pt = GetPeriodicTable()
    elements = [pt.GetElementSymbol(i) for i in atom_types]
    elements = [e for e in elements if MolFromSmiles(e) is not None]
    logger.info(f'Using {len(elements)} elements: {elements}')

    # Make the reward function
    if args.reward == 'ic50':
        model = load_model(os.path.join(mpnn_dir, 'model.h5'), custom_objects=custom_objects)
        with open(os.path.join(mpnn_dir, 'bond_types.json')) as fp:
            bond_types = json.load(fp)
        reward = MPNNReward(model, atom_types, bond_types, maximize=False)
    elif args.reward == 'logP':
        reward = LogP(maximize=False)
    else:
        raise ValueError(f'Reward function not defined: {args.reward}')

    # Set up environment
    action_space = MoleculeActions(elements, allow_removal=not args.no_backtrack)
    init_mol = args.initial_molecule
    if init_mol is not None:
        init_mol = convert_smiles_to_nx(init_mol)
    env = Molecule(action_space, max_steps=args.max_steps, reward=reward,
                   init_mol=init_mol)
    logger.debug('using environment: %s' % env)

    # Setup agent
    agent = DQNFinalState(env, preprocessor=MorganFingerprints(), epsilon=args.epsilon,
                          q_network_dense=args.hidden_layers)

    # Make a test directory
    test_dir = os.path.join('rl_tests', datetime.now().isoformat().replace(":", "."))
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    # Write the test parameters to the test directory
    with open(os.path.join(test_dir, 'config.json'), 'w') as fp:
        json.dump(run_params, fp)

    # Run experiment
    with open(os.path.join(test_dir, 'molecules.csv'), 'w', newline='') as log_fp:
        log_file = DictWriter(log_fp, fieldnames=['episode', 'step', 'epsilon',
                                                  'smiles', 'reward', 'loss'])
        log_file.writeheader()

        start = timeit.default_timer()
        run_experiment(args.episodes, args.max_steps, args.q_update_freq, log_file)
        end = timeit.default_timer()

        # Save the performance information
        platform_info = get_platform_info()
        platform_info['runtime'] = end - start
        with open(os.path.join(test_dir, 'performance.json'), 'w') as fp:
            json.dump(platform_info, fp)
