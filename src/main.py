
""" Dataset Format, set of possible states, set of possible observations (python set)
Input: data - a list of sequences, each a list of (state, observation) tuples

Initialize empty list all_states = []
Initialize empty list all_observations = []

For each sequence in data:
    Initialize empty list seq_states = []
    Initialize empty list seq_observations = []
    
    For each (state, observation) in sequence:
        Append state to seq_states
        Append observation to seq_observations
    
    Append seq_states to all_states
    Append seq_observations to all_observations

Convert all_states to NumPy array: states_array = np.array(all_states)
Convert all_observations to NumPy array: observations_array = np.array(all_observations)

Output: states_array, observations_array
"""

def main():
    # TODO: function to calculate emission parameters (labelled_data) -> dictionary of state->observation
    # TODO: function to calculate transition parameters (labelled_data) -> dictionary of state->state
    pass