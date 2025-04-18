from typing import Dict, List
import numpy as np

from utils import _safe_log


def viterbi(
    transition_probabilities: Dict[str, Dict[str, float]],
    emission_probabilities: Dict[str, Dict[str, float]],
    observation_sequence: List[str],
):
    # Extract initial and final probabilities internally
    initial_probabilities = transition_probabilities.get("START", {})
    final_probabilities = {
        tag: transition_probabilities[tag].get("STOP", 0.0)
        for tag in transition_probabilities
        if tag != "START" and tag != "STOP"
    }

    states = list(transition_probabilities.keys())
    states = [s for s in states if s not in ("START", "STOP")]
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
        if from_state not in state_to_idx:
            continue
        for to_state, prob in to_probs.items():
            if to_state not in state_to_idx:
                continue
            A[state_to_idx[from_state]][state_to_idx[to_state]] = prob

    # Emission matrix B[u][x] is probability of u emitting observation x
    B = np.zeros((N, V))
    for state, obs_probs in emission_probabilities.items():
        for obs, prob in obs_probs.items():
            if state in state_to_idx and obs in obs_to_idx:
                B[state_to_idx[state]][obs_to_idx[obs]] = prob

    pi = np.zeros((n, N))
    backpointer = np.zeros((n, N), dtype=int)

    # FORWARD PASS
    # base case: pi(0, START) = 1 so only consider paths from START for the first state i=1
    for s in range(N):
        state = idx_to_state[s]
        x_1 = observation_sequence[0]
        x_1_idx = obs_to_idx.get(x_1, obs_to_idx.get("#UNK#", -1))

        transition = initial_probabilities.get(state, 0.0)
        emission = B[s][x_1_idx] if x_1_idx != -1 else 0.0

        pi[0][s] = _safe_log(transition) + _safe_log(emission)
        backpointer[0][s] = 0

    # dp j=(1, ... n)
    for j in range(1, n):
        x_j = observation_sequence[j]
        x_j_idx = obs_to_idx.get(x_j, obs_to_idx.get("#UNK#", -1))

        for s in range(N):
            # take the max of the array that contains pi[ps][s]*A[ps][s]*B[s][x_j_idx]
            max_prob = -np.inf
            best_prev_s = 0
            # can maximize accross transitions*pi(j-1) since emissions are independent of previous state
            for ps in range(N):
                prob = pi[j - 1][ps] + _safe_log(A[ps][s])
                if prob > max_prob:
                    max_prob = prob
                    best_prev_s = ps

            emission = B[s][x_j_idx] if x_j_idx != -1 else 0.0
            pi[j][s] = max_prob + _safe_log(emission)
            backpointer[j][s] = best_prev_s

    # final step:
    pi_j = np.zeros(N)
    for s in range(N):
        state = idx_to_state[s]
        transition = final_probabilities.get(state, 0.0)
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


def generate_viterbi_output(test_data, transition_probs, emission_probs):
    with open("../EN/dev.p2.out", "w") as f:
        for sentence in test_data:
            tag_sequence = viterbi(transition_probs, emission_probs, sentence)
            for word, tag in zip(sentence, tag_sequence[1:-1]):  # strip START/STOP
                f.write(f"{word} {tag}\n")
            f.write("\n")
