from typing import Dict, List


def viterbi(
    transition_probabilities: Dict[str, Dict[str, float]],
    emission_probabilities: Dict[str, Dict[str, float]],
    observation_sequence: List[str],
    initial_probabilities: Dict[str, float],
    final_probabilities: Dict[str, float],
    top_k=1,
):
    # transition is transition_probabilities[state i-1]= {state 0 to n each with a float}
    # emission is emission_probabilities[state i] = {observation 0 to n each with a float}

    # maybe also input the initial probabilities (START->y) and final probabilities (STOP->y)
    # determine whether this is part of the transition probabiliteis or not

    state_keys = transition_probabilities.keys()
    N = len(state_keys)

    # Get the keys from the first state's emission dictionary
    first_state = next(iter(emission_probabilities))
    observation_keys = emission_probabilities[first_state].keys()
    V = len(observation_keys)

    n = len(observation_sequence)

    dp = [0] * N

    # extract the set of transition probabilities
    # convert to numpy matrices for each
    # create a dp table (forward pass)
    # reconstruct the best path (list of states) should be able to later store top-k
