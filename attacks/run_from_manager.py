import numpy as np
import os
import time
import multiprocessing
from experiment import run_experiment
import pickle
import pytz
from utils import ExpParams
import argparse


def print_exp_to_run(d, n_runs=-1, exp_id=-1):
    if exp_id > 0:
        print('***********************[ {:d} ]***********************'.format(exp_id))
    else:
        print('********************************************************')
    print('Setting: {:s} {:s} nkw={:d}'.format(d['dataset'], str(d['gen_p']), d['nkw']))
    print('Query params: {:s}'.format(str(d['query_p'])))
    print('Def: {:s} {:s}'.format(d['def'], str(d['def_p'])))
    print('Att: {:s} {:s}'.format(d['att'], str(d['att_p'])))
    if n_runs > 0:
        print("* Number of runs: {:d}".format(n_runs))
    print('********************************************************', flush=True)


def run_and_save_experiment_all_together(exp_params, exp_id, seed, results_path):

    # We only print the full experiments when we launch the first experiment of this series:
    if seed == 0:
        # time.sleep(np.random.randint(0, 3))  # Some delay to avoid printing overlaps
        print('')
        print_exp_to_run(exp_params, exp_id=exp_id)
    else:
        time.sleep(0.1)
    print('[{:d}-{:d}]'.format(exp_id, seed), end='', flush=True)

    dir_to_save = os.path.join(results_path, 'done_runs_{:04}'.format(exp_id))
    while not os.path.exists(dir_to_save):
        try:
            os.makedirs(dir_to_save)
        except FileExistsError:
            print("Directory {:s} already existed, gonna sleep a bit".format(dir_to_save), flush=True)
            time.sleep(np.random.randint(0, 3))
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print("Could not create directory {:s}, gonna sleep a bit".format(dir_to_save), flush=True)
            time.sleep(np.random.randint(0, 3))

    temp_filename = 'results_{:03d}.temp'.format(seed)
    res_filename = 'results_{:03d}.pkl'.format(seed)
    if (os.path.exists(os.path.join(dir_to_save, temp_filename)) or os.path.exists(os.path.join(dir_to_save, res_filename))):
        print("[{:d}-{:d} already existed!]".format(exp_id, seed), end='', flush=True)
        return -1
    else:
        with open(os.path.join(dir_to_save, temp_filename), 'w') as f:
            pass

        exp_params_object = ExpParams(exp_params)
        run_experiment(exp_params_object, seed, debug_mode=False)
        os.remove(os.path.join(dir_to_save, temp_filename))
        print("[!{:d}-{:d}]".format(exp_id, seed), end='', flush=True)
        return 1


PRO_DATASETS_PATH = 'datasets_pro'
EXPERIMENTS_PATH = 'results'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('manager_number', type=int, nargs='?', default=1)
    args = parser.parse_args()

    manager_filename = 'manager_data{:d}.pkl'.format(args.manager_number)
    if not os.path.exists(manager_filename):
        raise ValueError("File {:s} does not exist!".format(manager_filename))
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)
        print("Loaded {:s}".format(manager_filename))

    time_init = time.time()
    results_path = 'results'
    tz_ON = pytz.timezone('Canada/Eastern')

    params_list = []
    for index, row in manager.experiments.iterrows():
        exp_dict = row.to_dict()
        del exp_dict['target_runs']
        del exp_dict['res_pointer']
        exp_id = row['res_pointer']
        for seed in range(row['target_runs']):
            if seed not in manager.results[exp_id]['seed'].values:
                params_list.append((exp_dict.copy(), exp_id, seed, results_path))

    print("Done gathering experiments, we have a total of {:d} seeds:".format(len(params_list)), flush=True)
    for _, exp_id, seed, _ in params_list:
        print("({:d}-{:d})".format(exp_id, seed), end='', flush=True)
    print("")
    # pool = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
    pool = multiprocessing.Pool(140)
    pool.starmap(run_and_save_experiment_all_together, params_list, chunksize=1)
    print("DONE!!!!!")
