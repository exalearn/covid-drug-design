import os
import json
import timeit
import logging
import platform
import tensorflow as tf
import keras.backend as K
from tqdm import tqdm
from datetime import datetime
from csv import DictWriter
from rdkit import RDLogger
from argparse import ArgumentParser
from molgym.agents.moldqn import DQNFinalState
from molgym.agents.preprocessing import MorganFingerprints
from molgym.envs.simple import Molecule

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
    best_reward = 0

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
            log_file.writerow({'episode': e, 'step': s, 'smiles': env.state, 'loss': loss,
                               'reward': reward, 'epsilon': agent.epsilon})

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
    arg_parser.add_argument('--max_steps', help='Maximum number of steps per episode',
                            default=32, type=int)
    arg_parser.add_argument('--episodes', help='Number of episodes to run',
                            default=200, type=int)
    arg_parser.add_argument('--q-update-freq', help='After how many episodes to update Q network',
                            default=10, type=int)

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__

    # Setup Keras/Tensorflow
    sess = tf.Session()
    K.set_session(sess)

    # Set up environment
    env = Molecule(max_steps=args.max_steps)
    logger.debug('using environment: %s' % env)

    # Setup agent
    agent = DQNFinalState(env, preprocessor=MorganFingerprints(), epsilon=args.epsilon)

    # Make a test directory
    test_dir = os.path.join('rl_tests', str(int(datetime.now().timestamp())))
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
