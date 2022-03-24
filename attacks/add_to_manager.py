from manager import Manager
from exp_params import ExpParams
import os
import pickle
import pandas as pd
import numpy as np


if __name__ == "__main__":

    manager_filename = 'manager_data_pancake9.pkl'
    if not os.path.exists(manager_filename):
        manager = Manager()
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)

    category_list = ['privacy', 'politics', 'cryptography', 'activism', 'security']

    exp_params = ExpParams()
    exp_params.set_defense_params('none')

    # for nq in [int(100e3), int(500e3)]:
    for month in [1, 7]:
        for category in category_list:
            exp_params.set_generation_params('wiki', category=category, language='en', year=2020, month=7, month2=7 + 5)
            for nq in [int(100e3), int(500e3)]:
                exp_params.set_nq(nq)
                # for month in [1, 3, 5, 7]:
                # for niters in [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
                # for niters in [500]:
                #     exp_params.set_attack_params('itersap', pfree=0.25, niters=niters, ver='v1.1', year=2020, month=month, month2=month+5)
                #     manager.initialize_or_add_runs(exp_params, target_runs=30)
                exp_params.set_attack_params('hungfreq', year=2020, month=month, month2=month+5)
                manager.initialize_or_add_runs(exp_params, target_runs=30)

    print("Saving ResultsManager...")
    with open(manager_filename, 'wb') as f:
        pickle.dump(manager, f)
