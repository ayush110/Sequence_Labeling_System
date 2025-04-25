from collections import (
    Counter,
    defaultdict,
)  # https://docs.python.org/3/library/collections.html#collections.Counter
from typing import Dict, List, Tuple

def compute_emission_parameters(training_data: List[List[Tuple[str, str]]], k: int = 3) -> Dict[str, Dict[str, float]]:
    """
    Computes emission parameters using Laplace smoothing.

    Returns:
        emission_probabilities: dict[state][observation] = P(observation | state)
    """
    # count total observations
    observation_counts = Counter(
        obs for sequence in training_data for obs, _ in sequence
    )

    # replace rare words (k < 3) with #UNK# in the dataset
    processed_data = []
    for sequence in training_data:
        new_seq = []
        for obs, state in sequence:
            if observation_counts[obs] < k:
                obs = "#UNK#"
            new_seq.append((obs, state))
        processed_data.append(new_seq)

    # re-count states and state->observations after replacing with #UNK#
    state_counts = Counter()
    state_obs_counts = Counter()
    vocabulary = set()

    for sequence in processed_data:
        for obs, state in sequence:
            state_counts[state] += 1
            state_obs_counts[(state, obs)] += 1
            vocabulary.add(obs)

    V = len(vocabulary)  # total number of unique observations (including #UNK#)

    # initialize emission probabilities
    emission_probabilities = defaultdict(dict)
    for state in state_counts:
        for obs in vocabulary:
            # calculate emission probability with Laplace smoothing
            # P(observation | state) = (count(state, observation) + 1) / (count(state) + V)
            count = state_obs_counts.get((state, obs), 0)
            emission_probabilities[state][obs] = (count + 1) / (state_counts[state] + V)

    return emission_probabilities


def generate_tags(emission_probabilities: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    Generates a tag for each observation (word) based on the emission probabilities.
    """
    # create a dictionary mapping each observation to its most likely state
    observation_to_tag = {
        obs: max(
            emission_probabilities,
            key=lambda state: emission_probabilities[state].get(obs, 0),
        )
        for obs in set(
            obs
            for state_probs in emission_probabilities.values()
            for obs in state_probs
        )
    }
    return observation_to_tag


def emission_predict(sentence: List[str], emission_probs: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Predict tags based only on emission probabilities (argmax over states).
    """
    # create a set of known words from the emission probabilities
    # and add #UNK# to the vocabulary
    known_words = set(word for probs in emission_probs.values() for word in probs)
    vocabulary = known_words.union({"#UNK#"})

    tags = []
    for word in sentence:
        obs = word if word in vocabulary else "#UNK#"
        # find the most likely tag for the observation (choose the maximum emission probability)
        best_tag = max(emission_probs, key=lambda tag: emission_probs[tag].get(obs, 0))
        tags.append(best_tag)
    return tags


# Example Usage:
# training_data = get_sequences_dataset("../EN/train")
# test_data = parse_test_data("../EN/dev.in")
# emission_parameters = compute_emission_parameters(training_data)
# tags = generate_tags(emission_parameters)

# generate_output(tags, test_data)
