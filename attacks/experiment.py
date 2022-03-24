import os
import numpy as np
import pickle
import time
import attacks
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import utils
from defense import apply_defense

PRO_DATASET_PATH = './datasets_pro/'


def load_pro_dataset(dataset_name):
    full_path = os.path.join(PRO_DATASET_PATH, dataset_name + '.pkl')
    if not os.path.exists(full_path):
        raise ValueError("The file {} does not exist".format(full_path))

    with open(full_path, "rb") as f:
        dataset, keyword_dict = pickle.load(f)

    return dataset, keyword_dict


def generate_keyword_queries(mode_query, frequencies, query_params):

    nqr = query_params['nqr']
    nkw = frequencies.shape[0]
    if mode_query == 'iid':
        assert frequencies.ndim == 1
        queries = list(np.random.choice(list(range(nkw)), nqr, p=frequencies))
    elif mode_query == 'markov':
        assert frequencies.ndim == 2
        ss = utils.get_steady_state(frequencies)
        queries = np.zeros(nqr, dtype=int)
        queries[0] = np.random.choice(len(range(nkw)), p=ss)
        for i in range(1, nqr):
            queries[i] = np.random.choice(len(range(nkw)), p=frequencies[:, queries[i - 1]])
    elif mode_query == 'each':
        queries = [list(np.random.permutation(nkw))[:min(nqr, nkw)]]
    else:
        raise ValueError("Frequencies has {:d} dimensions, only 1 or 2 allowed".format(frequencies.ndim))
    return queries


def build_frequencies_from_file(dataset_name, chosen_keywords, keyword_dict, mode_fs):

    nkw = len(chosen_keywords)
    if dataset_name in ('enron', 'lucene'):
        freq_weeks = np.array([keyword_dict[kw]['trend'] for kw in chosen_keywords])
        for i_col in range(freq_weeks.shape[1]):
            if sum(freq_weeks[:, i_col]) == 0:
                print("The {:d}th column of the trend matrix adds up to zero, making it uniform!".format(i_col))
                freq_weeks[:, i_col] = 1 / nkw
            else:
                freq_weeks[:, i_col] = freq_weeks[:, i_col] / sum(freq_weeks[:, i_col])

        if mode_fs == 'same': # Take last year of data
            freq_cli = freq_real = freq_adv = np.mean(freq_weeks[:, -52:], axis=1)
        elif mode_fs == 'past': # First half of year for adv, last half is real and client's
            freq_adv = np.mean(freq_weeks[:, -52:-26], axis=1)
            freq_cli = freq_real = np.mean(freq_weeks[:, -26:], axis=1)
        else:
            raise ValueError("Frequencies split mode '{:s}' not allowed for {:s}".format(mode_fs, dataset_name))
    elif dataset_name.startswith('wiki'):
        category = dataset_name.split('-')[1]
        # TODO HOW ARE WE GOING TO SAVE THESE DATASETS?
        freq_adv, freq_cli, freq_real = np.zeros((nkw, nkw))
        raise NotImplementedError()
    else:
        raise ValueError("Dataset {:s} not allowed".format(dataset_name))
    return freq_adv, freq_cli, freq_real


def generate_train_test_data(gen_params):
    nkw = gen_params['nkw']
    dataset_name = gen_params['dataset']
    mode_kw = gen_params['mode_kw']
    mode_ds = gen_params['mode_ds']
    freq_name = gen_params['freq']
    mode_fs = gen_params['mode_fs']

    dataset, keyword_dict = load_pro_dataset(dataset_name)

    if mode_kw == 'top':
        chosen_keywords = sorted(keyword_dict.keys(), key=lambda x: keyword_dict[x]['count'], reverse=True)[:nkw]
    elif mode_kw == 'rand':
        keywords = list(keyword_dict.keys())
        permutation = np.random.permutation(len(keywords))
        chosen_keywords = [keywords[idx] for idx in permutation[:nkw]]
    else:
        raise ValueError("Keyword selection mode '{:s}' not allowed".format(mode_kw))

    if mode_ds.startswith('same'):
        percentage = 50 if mode_ds == 'same' else int(mode_ds[4:])
        assert 0 < percentage <= 100
        permutation = np.random.permutation(len(dataset))
        dataset_selection = [dataset[i] for i in permutation[int(len(dataset) * percentage / 100):]]
        data_adv = dataset_selection
        data_cli = dataset_selection
    elif mode_ds.startswith('common'):
        percentage = 50 if mode_ds == 'common' else int(mode_ds[6:])
        assert 0 < percentage <= 100
        permutation = np.random.permutation(len(dataset))
        dataset_selection = [dataset[i] for i in permutation[int(len(dataset) * percentage / 100):]]
        data_adv = dataset_selection
        data_cli = dataset
    elif mode_ds == 'split':
        percentage = 50 if mode_ds == 'split' else int(mode_ds[5:])
        assert 0 < percentage < 100
        permutation = np.random.permutation(len(dataset))
        data_adv = [dataset[i] for i in permutation[int(len(dataset) * percentage / 100):]]
        data_cli = [dataset[i] for i in permutation[:int(len(dataset) * (100 - percentage) / 100)]]
    else:
        raise ValueError("Dataset split mode '{:s}' not allowed".format(mode_ds))

    if freq_name == 'file':
        freq_adv, freq_cli, freq_real = build_frequencies_from_file(dataset_name, keyword_dict, mode_fs)
    elif freq_name.startswith('zipf'):
        shift = int(mode_fs[5:]) if mode_fs.startswith('zipfs') else 0  # zipfs200 is a zipf with 200 shift
        aux = np.array([1 / (i + shift + 1) for i in range(nkw)])
        freq_adv = freq_cli = freq_real = aux / np.sum(aux)
    else:
        raise ValueError("Frequency name '{:s}' not implemented yet".format(freq_name))

    full_data_adv = {'dataset': data_adv,
                     'keywords': chosen_keywords,
                     'frequencies': freq_adv}
    full_data_client = {'dataset': data_cli,
                        'keywords': chosen_keywords,
                        'frequencies': freq_cli}
    return full_data_adv, full_data_client, freq_real


def run_attack(attack_name, **kwargs):
    if attack_name == 'freq':
        return attacks.freq_attack(**kwargs)
    elif attack_name == 'sap':
        return attacks.sap_attack(**kwargs)
    else:
        raise ValueError("Attack name '{:s}' not recognized".format(attack_name))


def run_experiment(exp_param, seed, debug_mode=False, exp_number=-1):

    np.random.seed(seed)
    full_data_adv, full_data_client, freq_real = generate_train_test_data(exp_param.gen_params)
    real_queries = generate_keyword_queries(exp_param.gen_params['mode_query'], freq_real, exp_param.query_params)
    observations, bw_overhead = apply_defense(full_data_client, exp_param.query_params, exp_param.def_params, real_queries)

    keyword_predictions_for_each_query = run_attack(exp_param.att_params['name'], obs=observations, aux=full_data_adv,
                                                    att_params=exp_param.att_params, def_params=exp_param.def_params)

    # Compute accuracy
    return -1, -1
    # if query_predictions_for_each_obs is None:
    #     accuracy, accuracy_un = np.nan, np.nan
    # elif multiple_results:
    #     niter_list = exp_param.att_params['niter_list']
    #     del exp_param.att_params['niter_list']
    #     flat_real = [kw for week_kws in real_queries for kw in week_kws]
    #     accuracy_list = []
    #     accuracy_un_list = []
    #     for query_pred_obs in query_predictions_for_each_obs:
    #         flat_pred = [kw for week_kws in query_pred_obs for kw in week_kws]
    #         acc_vector = np.array([1 if real == prediction else 0 for real, prediction in zip(flat_real, flat_pred)])
    #         accuracy = np.mean(acc_vector)
    #         accuracy_un = np.mean(np.array([np.mean([acc_vector[k] for k in range(len(flat_real)) if flat_real[k] == i]) for i in set(flat_real)]))
    #         accuracy_list.append(accuracy)
    #         accuracy_un_list.append(accuracy_un)
    #
    #     if debug_mode:
    #         for i, niters in enumerate(niter_list):
    #             print("niters={:d}. For {:s} vs {:s}, acc={:.3f} ({:.3f} unw), time={:.3f}, bw={:.3f} | {} {}"
    #                   .format(niters, exp_param.att_params['name'], exp_param.def_params['name'], accuracy_list[i], accuracy_un_list[i],
    #                           attack.time_info['time_attack'], bw_overhead, exp_param.att_params, exp_param.def_params))
    #         os.remove(os.path.join(experiments_path_ext, 'results_{:06d}.temp'.format(exp_number)))
    #         return accuracy_list[-1], accuracy_un_list[-1]
    #     else:
    #         for niters, accuracy, accuracy_un in zip(niter_list, accuracy_list, accuracy_un_list):
    #             results_filename = 'results_{:06d}_{:d}.pkl'.format(exp_number, niters)
    #             exp_param.att_params['niters'] = niters
    #             with open(os.path.join(experiments_path_ext, results_filename), 'wb') as f:
    #                 res_dict = {'seed': seed, 'accuracy': accuracy, 'accuracy_un': accuracy_un}
    #                 time_info = attack.return_time_info()
    #                 res_dict.update(time_info)
    #                 res_dict['bw_overhead'] = bw_overhead
    #                 pickle.dump((exp_param, res_dict), f)
    #         os.remove(os.path.join(experiments_path_ext, 'results_{:06d}.temp'.format(exp_number)))
    #         return -1, -1
    #
    # else:
    #     flat_real = [kw for week_kws in real_queries for kw in week_kws]
    #     flat_pred = [kw for week_kws in query_predictions_for_each_obs for kw in week_kws]
    #     acc_vector = np.array([1 if real == prediction else 0 for real, prediction in zip(flat_real, flat_pred)])
    #     accuracy = np.mean(acc_vector)
    #     acc_un_vector = np.array([np.mean([acc_vector[k] for k in range(len(flat_real)) if flat_real[k] == i]) for i in set(flat_real)])
    #     accuracy_un = np.mean(acc_un_vector)
    #
    #     if debug_mode:
    #         print("For {:s} vs {:s}, acc={:.3f} ({:.3f} unw), time={:.3f}, bw={:.3f} | {} {}"
    #               .format(exp_param.att_params['name'], exp_param.def_params['name'], accuracy, accuracy_un, attack.time_info['time_attack'], bw_overhead,
    #                       exp_param.att_params, exp_param.def_params))
    #         os.remove(os.path.join(experiments_path_ext, 'results_{:06d}.temp'.format(exp_number)))
    #         return accuracy, accuracy_un
    #     else:
    #         results_filename = 'results_{:06d}.pkl'.format(exp_number)
    #         with open(os.path.join(experiments_path_ext, results_filename), 'wb') as f:
    #             res_dict = {'seed': seed, 'accuracy': accuracy, 'accuracy_un': accuracy_un}
    #             time_info = attack.return_time_info()
    #             res_dict.update(time_info)
    #             res_dict['bw_overhead'] = bw_overhead
    #             pickle.dump((exp_param, res_dict), f)
    #         os.remove(os.path.join(experiments_path_ext, 'results_{:06d}.temp'.format(exp_number)))
    #         return -1, -1

