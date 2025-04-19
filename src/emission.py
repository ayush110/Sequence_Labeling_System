from collections import (
    Counter,
    defaultdict,
)  # https://docs.python.org/3/library/collections.html#collections.Counter


def compute_emission_parameters(training_data, k=3):
    """
    Computes emission parameters using Laplace smoothing.

    Returns:
        emission_probabilities: dict[state][observation] = P(observation | state)
    """
    # Count total observations
    observation_counts = Counter(
        obs for sequence in training_data for obs, _ in sequence
    )

    # Replace rare words with #UNK# in the dataset
    processed_data = []
    for sequence in training_data:
        new_seq = []
        for obs, state in sequence:
            if observation_counts[obs] < k:
                obs = "#UNK#"
            new_seq.append((obs, state))
        processed_data.append(new_seq)

    # Re-count after replacing with #UNK#
    state_counts = Counter()
    state_obs_counts = Counter()
    vocabulary = set()

    for sequence in processed_data:
        for obs, state in sequence:
            state_counts[state] += 1
            state_obs_counts[(state, obs)] += 1
            vocabulary.add(obs)

    V = len(vocabulary)  # total number of unique observations (including #UNK#)

    # Initialize emission probabilities
    emission_probabilities = defaultdict(dict)
    for state in state_counts:
        for obs in vocabulary:
            count = state_obs_counts.get((state, obs), 0)
            emission_probabilities[state][obs] = (count + 1) / (state_counts[state] + V)

    return emission_probabilities


def generate_tags(emission_probabilities):
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


def emission_predict(sentence, emission_probs):
    """
    Predict tags based only on emission probabilities (argmax over states).
    """
    known_words = set(word for probs in emission_probs.values() for word in probs)
    vocabulary = known_words.union({"#UNK#"})

    tags = []
    for word in sentence:
        obs = word if word in vocabulary else "#UNK#"
        best_tag = max(emission_probs, key=lambda tag: emission_probs[tag].get(obs, 0))
        tags.append(best_tag)
    return tags


# Example Usage:
# training_data = get_sequences_dataset("../EN/train")
# test_data = parse_test_data("../EN/dev.in")
# emission_parameters = compute_emission_parameters(training_data)
# tags = generate_tags(emission_parameters)

# generate_output(tags, test_data)
