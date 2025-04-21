from typing import Dict, List
import numpy as np
import pandas as pd

from utils import _safe_log


def print_transition_matrix(A, states, initial_probs, final_probs):
    A = np.array(A)  # Convert to NumPy array for easier checks

    if A.ndim != 2:
        raise ValueError("Transition matrix must be 2-dimensional.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Transition matrix must be square.")
    if A.shape[0] != len(states):
        raise ValueError("Matrix dimensions must match number of states.")

    # Add initial and final transitions to matrix
    # Initial row for "START" (row 0) and final column for "STOP" (last column)
    transition_matrix = np.zeros(
        (len(states) + 2, len(states) + 2)
    )  # +2 for "START" and "STOP"

    # Fill in transition matrix with the internal transitions (A)
    transition_matrix[1:-1, 1:-1] = A

    # Fill in initial probabilities for "START" state (row 0)
    for i, state in enumerate(states):
        transition_matrix[0, i + 1] = initial_probs.get(state, 0)

    # Fill in final probabilities for "STOP" state (last row)
    for i, state in enumerate(states):
        transition_matrix[i + 1, -1] = final_probs.get(state, 0)

    # Optional: check row sums
    print("\nTransition Matrix:")
    df = pd.DataFrame(
        transition_matrix,
        index=["START"] + states + ["STOP"],
        columns=["START"] + states + ["STOP"],
    )
    print(df.round(4))

    # Check that non-"START" and non-"STOP" rows sum to 1.0
    print("\nRow Sums:")
    for i, state in enumerate(["START"] + states + ["STOP"]):
        row_sum = transition_matrix[i].sum()
        print(f"{state:>8}: sum = {row_sum:.4f}", end="")
        if state not in {"START", "STOP"} and not np.isclose(row_sum, 1.0, atol=1e-2):
            print("  Warning: Row does not sum to 1.0")
        else:
            print()


def print_emission_matrix(B, state_to_idx, obs_to_idx):
    """
    Prints the emission matrix B and checks for common issues.

    Args:
    B (np.ndarray): The emission matrix (N x V).
    state_to_idx (dict): Mapping of states to indices.
    obs_to_idx (dict): Mapping of observations to indices.
    """
    # Ensure B is a numpy array for easier checks
    B = np.array(B)

    # Check if the matrix is well-formed
    if B.ndim != 2:
        raise ValueError("Emission matrix must be 2-dimensional.")
    if B.shape[0] != len(state_to_idx):
        raise ValueError("Emission matrix row count must match the number of states.")
    if B.shape[1] != len(obs_to_idx):
        raise ValueError(
            "Emission matrix column count must match the number of observations."
        )

    # Optional: Normalize rows of B so they sum to 1
    row_sums = B.sum(axis=1, keepdims=True)
    B_normalized = B / row_sums  # Normalize each row

    # Print out the normalized matrix
    print("\nNormalized Emission Matrix (Rows sum to 1):")
    df_normalized = pd.DataFrame(
        B_normalized, index=list(state_to_idx.keys()), columns=list(obs_to_idx.keys())
    )
    print(df_normalized.round(4))

    # Check if the rows sum to 1.0
    print("\nRow Sums Check:")
    for state, row in zip(state_to_idx.keys(), B_normalized):
        row_sum = row.sum()
        print(f"{state:>10}: sum = {row_sum:.4f}", end="")
        if not np.isclose(row_sum, 1.0, atol=1e-2):
            print("  Warning: Row does not sum to 1.0")
        else:
            print()

    # Check for NaN or infinite values in the matrix
    print("\nInvalid Values Check:")
    if np.any(np.isnan(B_normalized)):
        print("Warning: Matrix contains NaN values.")
    if np.any(np.isinf(B_normalized)):
        print("Warning: Matrix contains infinite values.")
    else:
        print("Matrix is valid, no NaN or infinite values.")

    # Optionally: Calculate and print the most likely observation for each state
    print("\nMost Likely Observation for Each State:")
    most_likely = np.argmax(
        B_normalized, axis=1
    )  # Index of the max probability for each state
    for state, idx in state_to_idx.items():
        best_obs = list(obs_to_idx.keys())[most_likely[idx]]
        print(f"State {state}: Most likely observation is {best_obs}")


def print_viterbi_table(pi, backpointer, observation_sequence, idx_to_state, states):
    n = len(observation_sequence)
    print("\nViterbi Table (log-probabilities and backpointers):")
    for t in range(n):
        print(f"t={t}, Observation={observation_sequence[t]}")
        for s in range(len(states)):
            state = idx_to_state[s]
            print(
                f"  State={state}, Log-Probability={pi[t][s]}, Backpointer={backpointer[t][s]}"
            )
        print("-" * 30)


def viterbi(
    transition_probabilities: Dict[str, Dict[str, float]],
    emission_probabilities: Dict[str, Dict[str, float]],
    observation_sequence: List[str],
):
    """
    Implements the Viterbi algorithm for HMM sequence decoding.

    Args:
        transition_probabilities: Nested dict of transition probabilities q(y_i | y_{i-1})
        emission_probabilities: Nested dict of emission probabilities e(x_i | y_i)
        observation_sequence: List of observed words (tokens)

    Returns:
        List of most likely hidden states (tags)
    """

    # Extract initial (START → y) and final (y → STOP) transition probabilities
    # These are needed to initialize and terminate the DP correctly
    initial_probabilities = transition_probabilities.get("START", {})
    final_probabilities = {
        tag: transition_probabilities[tag].get("STOP", 0.0)
        for tag in transition_probabilities
        if tag not in ("START", "STOP")
    }

    # transition is transition_probabilities[state i-1]= {state 0 to n each with a float}
    # emission is emission_probabilities[state i] = {observation 0 to n each with a float}

    # Get list of actual states, excluding START and STOP
    states = [s for s in transition_probabilities if s not in ("START", "STOP")]
    state_to_idx = {s: i for i, s in enumerate(states)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    # Get list of all observation types
    first_state = next(iter(emission_probabilities))
    observations = list(emission_probabilities[first_state].keys())
    obs_to_idx = {o: i for i, o in enumerate(observations)}

    N = len(states)
    V = len(observations)
    n = len(observation_sequence)

    # Transition matrix A[v][u] = P(u | v), i.e., from v (previous) to u (current)
    A = np.zeros((N, N))
    for from_state, to_probs in transition_probabilities.items():
        if from_state not in state_to_idx:
            continue  # skip START/STOP
        for to_state, prob in to_probs.items():
            if to_state not in state_to_idx:
                continue
            A[state_to_idx[from_state]][state_to_idx[to_state]] = prob

    # Emission matrix B[u][x] = P(x | u), i.e., probability of state u emitting observation x
    B = np.zeros((N, V))
    for state, obs_probs in emission_probabilities.items():
        for obs, prob in obs_probs.items():
            if state in state_to_idx and obs in obs_to_idx:
                B[state_to_idx[state]][obs_to_idx[obs]] = prob

    # print_transition_matrix(A, states, initial_probabilities, final_probabilities)
    # print_emission_matrix(B, state_to_idx, obs_to_idx)

    # Initialize DP table for max log-probabilities
    pi = np.zeros((n, N))  # pi[j][s] = best score for reaching state s at time j
    backpointer = np.zeros(
        (n, N), dtype=int
    )  # tracks best previous state for reconstruction

    # FORWARD PASS
    # base case: pi(0, START) = 1 so only consider paths from START for the first state i=1
    for s in range(N):
        state = idx_to_state[s]
        x_1 = observation_sequence[0]
        x_1_idx = obs_to_idx.get(
            x_1, obs_to_idx.get("#UNK#", -1)
        )  # use UNK token if not found

        # transition from START → state and emission of first observation
        transition = initial_probabilities.get(state, 0.0)
        emission = B[s][x_1_idx] if x_1_idx != -1 else 0.0

        # log(P(y_1 | START)) + log(P(x_1 | y_1))
        pi[0][s] = _safe_log(transition) + _safe_log(emission)
        backpointer[0][s] = 0  # no real backpointer at step 0

        print(
            f"t=0, state={state}, transition={transition}, emission={emission}, pi={pi[0][s]}"
        )

    # dp j=(1, ... n)
    for j in range(1, n):
        x_j = observation_sequence[j]
        x_j_idx = obs_to_idx.get(x_j, obs_to_idx.get("#UNK#", -1))

        for s in range(N):
            max_prob = -np.inf
            best_prev_s = 0
            for ps in range(N):
                # can maximize across transitions * pi(j-1), since emissions are independent of previous state
                prob = pi[j - 1][ps] + _safe_log(A[ps][s])
                if prob > max_prob:
                    max_prob = prob
                    best_prev_s = ps

            # add log emission probability
            emission = B[s][x_j_idx] if x_j_idx != -1 else 0.0
            pi[j][s] = max_prob + _safe_log(emission)
            backpointer[j][s] = best_prev_s

            print(
                f"t={j}, curr_state={idx_to_state[s]}, prev_state={idx_to_state[best_prev_s]}, "
                f"max_prob={max_prob}, emission={emission}, pi={pi[j][s]}"
            )

    # FINAL STEP: include transition to STOP
    pi_j = np.zeros(N)
    for s in range(N):
        state = idx_to_state[s]
        transition = final_probabilities.get(state, 0.0)
        # pi(n+1, STOP) = max over all paths that end in state s and transition to STOP
        pi_j[s] = _safe_log(transition) + _safe_log(pi[-1][s])
        print(
            f"Final transition from state={state} to STOP, prob={transition}, pi_j={pi_j[s]}"
        )

    top_1_final_s = np.argmax(pi_j)
    # print(
    #     f"Top-1 final state index: {top_1_final_s}, state: {idx_to_state[top_1_final_s]}"
    # )

    top_1_path = [top_1_final_s]
    for i in range(n - 1, -1, -1):
        top_1_path.insert(0, backpointer[i][top_1_path[0]])
        # print(
        #     f"Backtrack step {i}, state index={top_1_path[0]}, state={idx_to_state[top_1_path[0]]}"
        # )

    # Decode list of state indices into actual state labels
    decoded_states = [idx_to_state[i] for i in top_1_path]
    print(f"Decoded sequence: {decoded_states}")
    print_viterbi_table(pi, backpointer, observation_sequence, idx_to_state, states)

    return decoded_states


def viterbi_predict(sentence, transition_probs, emission_probs):
    """
    Returns a sequence of tags for a given sentence using Viterbi algorithm.
    Replaces unknown words with #UNK#.
    """
    known_words = set(word for probs in emission_probs.values() for word in probs)
    mapped_sentence = [word if word in known_words else "#UNK#" for word in sentence]

    tag_sequence = viterbi(transition_probs, emission_probs, mapped_sentence)
    return tag_sequence[1:-1]  # Remove START and STOP
