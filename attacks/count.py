import numpy as np
from processing.process_obs import process_traces, compute_Vobs
from processing.process_aux import get_Vexp
from collections import Counter
import itertools

# TODO: THIS WAS COPIED FROM THE SAP PAPER, IT DOES NOT WORK IN THIS NEW FRAMEWORK YET!

def count_disambiguations(tags0, kws0, candidate_keywords_per_tag, n_kws, n_tags):
    """Receives an initial (fixed) assignment of tags to keywords and computes how many disambiguations are solved
    current_map is a dictionary mapping tag to sets of candidate keywords"""

    current_map = candidate_keywords_per_tag.copy()
    for tag_id, kw_id in zip(tags0, kws0):
        current_map[tag_id] = [kw_id]

    ambiguous_tags = [tag_id for tag_id, candidates in current_map.items() if len(candidates) > 1]
    known_tags = [tag_id for tag_id, candidates in current_map.items() if len(candidates) == 1]
    fresh_pass = False
    while not fresh_pass and len(ambiguous_tags) > 1:
        fresh_pass = True
        for tag_id in ambiguous_tags:
            current_candidates = current_map[tag_id]
            new_candidates = []
            for candidate_kw_id in current_candidates:
                check = [m_matrix[current_map[known_tag][0]][candidate_kw_id] - window <= m_obs_matrix[known_tag][tag_id] <=
                         m_matrix[current_map[known_tag][0]][candidate_kw_id] + window for known_tag in known_tags]
                if all(check):
                    new_candidates.append(candidate_kw_id)

            if len(current_candidates) > len(new_candidates):
                fresh_pass = False
                current_map[tag_id] = new_candidates
                # print("  Removed {:d} candidates".format(len(current_candidates) - len(new_candidates)))

        for tag_id in ambiguous_tags:
            if len(current_map[tag_id]) == 1:
                ambiguous_tags.remove(tag_id)
                known_tags.append(tag_id)
                # print("  tag {:d} is disambiguated".format(tag_id))
                for tag_id_others in ambiguous_tags:
                    if tag_id != tag_id_others and current_map[tag_id][0] in current_map[tag_id_others]:
                        current_map[tag_id_others].remove(current_map[tag_id][0])
            elif len(current_map[tag_id]) == 0:
                # print("  Inconsistency!")
                return 0, None  # Inconsistency

    n_disambiguations = len(known_tags)

    cost_matrix = np.ones((n_kws, n_tags))
    for i_tag in range(n_tags):
        for candidate_kw_id in current_map[i_tag]:
            cost_matrix[candidate_kw_id, i_tag] = 0
    row_ind, col_ind = hungarian(cost_matrix)
    if cost_matrix[row_ind, col_ind].sum() > 0:
        # print("  There was no consistent matching!")
        return 0, None

    query_predictions_for_each_tag = {}
    for tag, keyword in zip(col_ind, row_ind):
        query_predictions_for_each_tag[tag] = keyword

    # print("  This matching has {:d} disambiguations, returning...".format(n_disambiguations))
    return n_disambiguations, query_predictions_for_each_tag



def _run_count_algorithm(Vexp, Vobs, window, tokens_by_popularity, nbrute_force=10):
    """Runs the generalized count attack using the brute-force method and Hoeffding bounds.
    Returns a dictionary that maps tag_ids to their assigned keywords (query_predictions_for_each_tag)
    Returns 0 instead if there is a global inconsistency

    :param Vexp:"""

    ntok = len(tokens_by_popularity)
    nkw = Vobs.shape[0]
    assert ntok >= nbrute_force

    # Build candidate sets per token
    candidate_kw_per_token = {}
    for tok_id in range(ntok):
        kw_list = [kw_id for kw_id in range(nkw) if Vexp[kw_id, kw_id] - window <= Vobs[tok_id, tok_id] <= Vexp[kw_id, kw_id] + window]
        if len(kw_list) == 0:
            # print("  tag_{:d} had zero candidates, aborting...".format(tag_id))
            return None
        candidate_kw_per_token[tok_id] = kw_list
    # print("LIST OF CANDIDATE KEYWORDS")
    # for tag_id in range(self.n_tags):
    #     print("{:d}: len={:d}".format(tag_id, len(candidate_keywords_per_tag[tag_id])))

    # Select brute-force sets to test
    candidate_sets_chosen = [candidate_kw_per_token[tok_id] for tok_id in tokens_by_popularity[:nbrute_force]]
    aux_combinations = list(itertools.product(*candidate_sets_chosen))
    all_combinations_to_test = [combination for combination in aux_combinations if len(combination) == len(set(combination))]
    if len(all_combinations_to_test) == 0:
        return None
    # print("There are {:d} combinations to test".format(len(all_combinations_to_test)))

    # Compute number of disambiguations in each of those sets
    test_results = [count_disambiguations(tokens_by_popularity[:nbrute_force], combination, candidate_kw_per_token, nkw, ntok)
                    for combination in all_combinations_to_test]

    test_results.sort(key=lambda x: x[0], reverse=True)

    # Choose output:
    if test_results[0][1] is not None:  # If one of these brute-forced matchings was feasible:
        # print("Found consistent mapping with {:d} disambiguated queries".format(test_results[0][0]))
        return test_results[0][1]
    else:
        # Ensure there is at least one feasible assignment with volumes
        cost_matrix = np.ones((nkw, ntok))
        for i_tag in range(ntok):
            for candidate_kw_id in candidate_kw_per_token[i_tag]:
                cost_matrix[candidate_kw_id, i_tag] = 0
        row_ind, col_ind = hungarian(cost_matrix)
        if cost_matrix[row_ind, col_ind].sum() > 0:
            return None
        else:
            feasible_assignment = {}
            for tag, keyword in zip(col_ind, row_ind):
                feasible_assignment[tag] = keyword
            return feasible_assignment


def count_attack(obs, aux, exp_params):
    """Count attack by Cash et al.

    Performs query recovery by following volume-based heuristics. Performs best with ground truth information.
    The original Paper: David Cash, Paul Grubbs, Jason Perry, and Thomas Ristenpart.
    Leakage-abuse attacks against searchable encryption.
    CCS 2015.

    :param dict obs: Attack observations
    :param dict aux: Attack auxiliary information
    :param dict exp_params: Experiment information, such as defense parameters
    :return: Keyword predictions for each query (a list of keyword id's)
    :rtype: dict
     """

    def_params = exp_params.def_params
    att_params = exp_params.att_params
    naive_flag = att_params['naive']
    pwindow = att_params['pwindow']  # Can be False or a float between 0 and 1
    token_trace, token_info = process_traces(obs, aux, def_params)
    ntok = len(token_info)
    nkw = len(aux['keywords'])
    ndocs = obs['ndocs']  # Integer
    nbrute_force = min(att_params['nbrute'], ntok)  # Integer
    Vobs = compute_Vobs(obs['trace_type'], token_info, ndocs)
    Vexp = get_Vexp(aux, exp_params.def_params, naive_flag)

    window = 0 if pwindow is False else np.sqrt(0.5 * np.log(2 / (1 - pwindow)) / ndocs)
    token_counter = Counter(token_trace)
    tokens_by_popularity = sorted(token_counter, key=token_counter.get, reverse=True)

    # TODO: process known data

    # Build candidate sets per token
    candidate_kw_per_token = {}
    for tok_id in range(ntok):
        kw_list = [kw_id for kw_id in range(nkw) if Vexp[kw_id, kw_id] - window <= Vobs[tok_id, tok_id] <= Vexp[kw_id, kw_id] + window]
        if len(kw_list) == 0:
            # print("  tag_{:d} had zero candidates, aborting...".format(tag_id))
            return None
        candidate_kw_per_token[tok_id] = kw_list
    # print("LIST OF CANDIDATE KEYWORDS")
    # for tag_id in range(self.n_tags):
    #     print("{:d}: len={:d}".format(tag_id, len(candidate_keywords_per_tag[tag_id])))

    # Select brute-force sets to test
    if nbrute_force > 0:
        candidate_sets_chosen = [candidate_kw_per_token[tok_id] for tok_id in tokens_by_popularity[:nbrute_force]]
        aux_combinations = list(itertools.product(*candidate_sets_chosen))
        all_combinations_to_test = [combination for combination in aux_combinations if len(combination) == len(set(combination))]
        if len(all_combinations_to_test) == 0:
            return None
        # print("There are {:d} combinations to test".format(len(all_combinations_to_test)))

        # Compute number of disambiguations in each of those sets
        test_results = [count_disambiguations(tokens_by_popularity[:nbrute_force], combination, candidate_kw_per_token, nkw, ntok)
                        for combination in all_combinations_to_test]

    test_results.sort(key=lambda x: x[0], reverse=True)



    keyword_predictions_for_each_token = _run_count_algorithm(Vexp, Vobs, window, tokens_by_popularity, nbrute)
    keyword_predictions_for_each_query = [keyword_predictions_for_each_token[token] for token in token_trace]
    return keyword_predictions_for_each_query
