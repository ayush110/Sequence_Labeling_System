

def viterbi_top_k(
    transition_probabilities: Dict[str, Dict[str, float]],
    emission_probabilities: Dict[str, Dict[str, float]],
    initial_probabilities: Dict[str, float],
    final_probabilities: Dict[str, float],
    observation_sequence: List[str],
    top_k=1,
):
    # transition is transition_probabilities[state i-1]= {state 0 to n each with a float}
    # emission is emission_probabilities[state i] = {observation 0 to n each with a float}

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

    # dp table for probabilities and backpointers
    # pi[i][j][k] stores the k-th best probability for state j at position i
    pi = np.zeros((n, N, top_k))
    # backpointer[i][j][k] stores the previous state for the k-th best path to state j at position i
    backpointer = np.zeros((n, N, top_k), dtype=int)

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
        pi[0][s][0] = transition * emission
        backpointer[0][s][0] = 0

    # dp j=(1, ... n)
    for j in range(1, n):
        x_j = observation_sequence[j]
        x_j_idx = obs_to_idx[x_j]

        for s in range(N):
            # For each state s, find top-k paths leading to it
            all_probs = []
            for ps in range(N):
                for k in range(top_k):
                    if pi[j-1][ps][k] > 0:  # Only consider valid paths
                        prob = pi[j-1][ps][k] * A[ps][s] * B[s][x_j_idx]
                        all_probs.append((prob, ps, k))
            
            # Sort all probabilities and take top-k
            all_probs.sort(reverse=True, key=lambda x: x[0])
            top_k_probs = all_probs[:top_k]
            
            # Store the top-k probabilities and their backpointers
            for k, (prob, prev_state, prev_k) in enumerate(top_k_probs):
                pi[j][s][k] = prob
                backpointer[j][s][k] = prev_state

    # final step:
    # Find top-k final states
    final_probs = []
    for s in range(N):
        state = idx_to_state[s]
        for k in range(top_k):
            if pi[-1][s][k] > 0:  # Only consider valid paths
                transition = final_probabilities[state]
                prob = transition * pi[-1][s][k]
                final_probs.append((prob, s, k))
    
    # Sort final probabilities and take top-k
    final_probs.sort(reverse=True, key=lambda x: x[0])
    top_k_final = final_probs[:top_k]

    # BACKWARD PASS (reconstruction)
    # Reconstruct the top-k paths
    top_k_paths = []
    for _, final_s, final_k in top_k_final:
        path = [final_s]
        for i in range(n-1, -1, -1):
            prev_state = backpointer[i][path[0]][final_k]
            path.insert(0, prev_state)
        top_k_paths.append(path)

    # Convert state indices to state names
    decoded_paths = [[idx_to_state[i] for i in path] for path in top_k_paths]
    return decoded_paths
