from typing import Dict, List, Tuple

import numpy as np


def _safe_log(x):
    return -np.inf if x <= 0 else np.log(x)


def viterbi(
    transition_probabilities: Dict[str, Dict[str, float]],
    emission_probabilities: Dict[str, Dict[str, float]],
    initial_probabilities: Dict[str, float],
    final_probabilities: Dict[str, float],
    observation_sequence: List[str],
):
    # transition is transition_probabilities[state i-1]= {state 0 to n each with a float}
    # emission is emission_probabilities[state i] = {observation 0 to n each with a float}

    # maybe also input the initial probabilities (START->y) and final probabilities (STOP->y)
    # determine whether this is part of the transition probabiliteis or not

    states = list(transition_probabilities.keys())  # this will be used for the pi table
    N = len(states)
    # Mapping states and observations to indices
    state_to_idx = {s: i for i, s in enumerate(states)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    # Get the keys from the first state's emission dictionary
    first_state = next(iter(emission_probabilities))
    observations = list(emission_probabilities[first_state].keys())
    V = len(observations)
    obs_to_idx = {o: i for i, o in enumerate(observations)}

    n = len(observation_sequence)

    # Transition matrix A[v][u] is probability of v -> u
    A = np.zeros((N, N))
    for from_state, to_probs in transition_probabilities.items():
        for to_state, prob in to_probs.items():
            A[state_to_idx[from_state]][state_to_idx[to_state]] = prob

    # Emission matrix B[u][x] is probability of u emitting observation x
    B = np.zeros((N, V))
    for state, obs_probs in emission_probabilities.items():
        for obs, prob in obs_probs.items():
            B[state_to_idx[state]][obs_to_idx[obs]] = prob

    # dp table, need to add start and final probabilities
    # pi(0, START) = 1 so only consider paths from START for the first state i=1
    # pi(n+1, STOP) = max
    # this will be for from state 1 to n
    pi = np.zeros((n, N))
    # backpointer will need to support k backpointers for top-k paths
    backpointer = np.zeros((n, N), dtype=int)

    # FORWARD PASS
    # base case: pi(0, START) = 1 so only consider paths from START for the first state i=1
    for s in range(N):
        state = idx_to_state[s]
        # first observation
        x_1 = observation_sequence[0]
        x_1_idx = obs_to_idx[x_1]

        # get a_START, state from initial_probabilities
        transition = initial_probabilities[state]
        emission = B[s][x_1_idx]

        # make each probability s transition probablity from start * emission(x)
        pi[0][s] = _safe_log(transition) + _safe_log(emission)

        backpointer[0][s] = 0

    # dp j=(1, ... n)
    for j in range(1, n):
        x_j = observation_sequence[j]
        x_j_idx = obs_to_idx[x_j]

        for s in range(N):
            # take the max of the array that contains pi[ps][s]*A[ps][s]*B[s][x_j_idx]
            max_prob = -np.inf  # _safe_log(0) = -inf
            best_prev_s = 0
            # can maximize accross transitions*pi(j-1) since emissions are independent of previous state
            for ps in range(N):
                prob = pi[j - 1][s] + _safe_log(A[ps][s])
                if prob > max_prob:
                    max_prob = prob
                    best_prev_s = ps

            backpointer[j][s] = best_prev_s

            emission = B[s][x_j_idx]
            pi[j][s] = max_prob + _safe_log(emission)

    # final step:
    pi_j = np.zeros(N)
    for s in range(N):
        state = idx_to_state[s]

        transition = final_probabilities[state]
        pi_j[s] = _safe_log(transition) + _safe_log(pi[-1][s])

    # BACKWARD PASS (reconstruction)
    # reconstruct the best path (list of states) should be able to later store top-k
    # best to stop is the best state from pi_j
    top_1_final_s = np.argmax(pi_j)

    top_1_path = [top_1_final_s]
    for i in range(n - 1, -1, -1):
        # find the best state
        top_1_path.insert(0, backpointer[i][top_1_path[0]])

    decoded_states = [idx_to_state[i] for i in top_1_path]
    return decoded_states


import heapq


def viterbi_top_k(
    transition_probabilities: Dict[str, Dict[str, float]],
    emission_probabilities: Dict[str, Dict[str, float]],
    initial_probabilities: Dict[str, float],
    final_probabilities: Dict[str, float],
    observation_sequence: List[str],
    top_k=1,
):
    states = list(transition_probabilities.keys())  # this will be used for the pi table
    state_to_idx = {s: i for i, s in enumerate(states)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    # Get the keys from the first state's emission dictionary
    first_state = next(iter(emission_probabilities))
    observations = list(emission_probabilities[first_state].keys())
    obs_to_idx = {o: i for i, o in enumerate(observations)}

    N = len(states)
    V = len(observations)
    n = len(observation_sequence)

    # Transition matrix A[v][u] is probability of v -> u
    A = np.zeros((N, N))
    for from_state, to_probs in transition_probabilities.items():
        for to_state, prob in to_probs.items():
            A[state_to_idx[from_state]][state_to_idx[to_state]] = prob

    # Emission matrix B[u][x] is probability of u emitting observation x
    B = np.zeros((N, V))
    for state, obs_probs in emission_probabilities.items():
        for obs, prob in obs_probs.items():
            B[state_to_idx[state]][obs_to_idx[obs]] = prob

    # dp table, need to add start and final probabilities
    # pi(0, START) = 1 so only consider paths from START for the first state i=1
    # pi(n+1, STOP) = max
    # this will be for from state 1 to n
    pi = [[[] for _ in range(N)] for _ in range(n)]
    # pi[j][s] will be a list of tuples (probability, path) for the top-k paths

    # each node in the dp table will store the top-k paths that led to that node (max-heap)
    # FORWARD PASS
    # base case: pi(0, START) = 1 so only consider paths from START for the first state i=1
    for s in range(N):
        state = idx_to_state[s]
        # first observation
        x_1 = observation_sequence[0]
        x_1_idx = obs_to_idx[x_1]

        # get a_START, state from initial_probabilities
        transition = initial_probabilities[state]
        emission = B[s][x_1_idx]

        # make each probability s transition probablity from start * emission(x)
        score = _safe_log(transition) + _safe_log(emission)
        pi[0][s].append(
            -score, [s]
        )  # store negative scores for maxheap (heapq is minheap)

    # dp j=(1, ... n)
    for j in range(1, n):
        x_j = observation_sequence[j]
        x_j_idx = obs_to_idx[x_j]

        for s in range(N):
            # take the top-k of the array that contains pi[ps][s]*A[ps][s]*B[s][x_j_idx]
            heap = []
            for ps in range(N):
                # add all top-k paths from the previous state
                for neg_score, path in pi[j - 1][ps]:
                    # score = prev_score + transition + emission
                    score = neg_score + _safe_log(A[ps][s]) + _safe_log(B[s][x_j_idx])
                    new_path = path + [s]
                    heap.append(-score, new_path)

            # get the top-k paths from the heap
            top_k_paths = heapq.nlargest(top_k, heap)
            pi[j][s] = top_k_paths

    # final step:
    # use a heap to store the net log probabilities of the top-k states for each final state
    final_heap = []
    for s in range(N):
        for neg_score, path in pi[-1][s]:
            # add stop transitions
            score = neg_score + _safe_log(final_probabilities[idx_to_state[s]])
            new_path = path + [s]
            final_heap.append(-score, new_path)

    # BACKWARD PASS (reconstruction) isnt needed since we are passing along the paths
    # get the top-k paths from the heap from largest to smallest (by negative score)
    top_k_paths = heapq.nlargest(top_k, final_heap)

    for neg_score, path in top_k_paths:
        # convert the state indices to state names
        path = [idx_to_state[i] for i in path]
        print(f"Path: {path}, Score: {-neg_score}")

    # since we are using a max-heap with negative scores, the k-th largest path will be the first element
    # imagine -0.1, -0.2, -0.3, -0.4, -0.5
    # 0.1 is the k-th largest path
    kth_largest_path = top_k_paths[0][1]
    decoded_kth_largest_path = [
        idx_to_state[i] for i in kth_largest_path
    ]  # convert the state indices to state names
    print(f"K-th largest path: {decoded_kth_largest_path}")
    return decoded_kth_largest_path
