from utils import get_sequences_dataset, parse_test_data
from collections import Counter # https://docs.python.org/3/library/collections.html#collections.Counter

def compute_emission_parameters(training_data, k = 3):
    """
    Computes the emission parameters from the training set using MLE.

    emission_probabilities = {'state': {'observation': probability,...},...}
    """
    # count all observations (words) in a single pass
    observation_counts = Counter(obs for sequence in training_data for obs, _ in sequence)

    # counters for states and state-observation pairs
    state_counts = Counter(state for sequence in training_data for _, state in sequence)
    state_observation_counts = Counter()

    # process training data while updating counts
    for sequence in training_data:
        for obs, state in sequence:
            # replace rare words with #UNK#
            if observation_counts[obs] < k:
                obs = "#UNK#"
            # update counts for state, (state, obs)
            state_observation_counts[(state, obs)] += 1

    emission_probabilities = {}
    # go through each observation
    for (state, obs), count in state_observation_counts.items():
        # if the state is not already in the emission probabilities,
        # add it with an empty dictionary
        if state not in emission_probabilities:
            emission_probabilities[state] = {}
        # calculate the probability of the observation given the state
        # using P(observation | state) = Count(state -> obs) / Count(state)
        # and store it in the emission probabilities dictionary 
        emission_probabilities[state][obs] = count / state_counts[state]

    return emission_probabilities

def generate_tags(emission_probabilities):
  """
  Generates a tag for each observation (word) based on the emission probabilities.
  """
  # create a dictionary mapping each observation to its most likely state
  observation_to_tag = {
    obs: max(emission_probabilities, key=lambda state: emission_probabilities[state].get(obs, 0))
    for obs in set(obs for state_probs in emission_probabilities.values() for obs in state_probs)
  }
  return observation_to_tag

def generate_output(tags, test_data):
  """
  Generates tag output in proper format in dev.p1.out
  """
  # open the output file in write mode
  with open("dev.p1.out", "w") as f:
    # write each observation and its corresponding tag
    for sample in test_data:
      for obs in sample:
        map_obs = obs
        if obs not in tags:
          # if the observation is not in the tags, use #UNK#
          map_obs = "#UNK#"
        # write the observation and its tag to the file
        tag = tags.get(map_obs)
        f.write(f"{obs} {tag}\n")
      f.write("\n")  # add a newline after each sample

# Example Usage:
# training_data = get_sequences_dataset("../EN/train")
# test_data = parse_test_data("../EN/dev.in")
# emission_parameters = compute_emission_parameters(training_data)
# tags = generate_tags(emission_parameters)

# generate_output(tags, test_data)
