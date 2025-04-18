from utils import get_sequences_dataset
from collections import Counter, defaultdict
from typing import Dict


def compute_transition_parameters(training_data: str) -> Dict[str, Dict[str, float]]:
    """
    Computes and returns transition probabilities q(y_i | y_{i-1}) in a nested dict format.

    Args:
        training_data (str): path to the training file (e.g., 'EN/train')

    Returns:
        Dict[str, Dict[str, float]]: transition probabilities {prev_tag: {curr_tag: prob}}
    """
    training_data

    transition_counts = Counter()
    tag_counts = Counter()

    for sentence in training_data:
        tags = ["START"] + [tag for _, tag in sentence] + ["STOP"]
        for i in range(1, len(tags)):
            prev_tag = tags[i - 1]
            curr_tag = tags[i]
            transition_counts[(prev_tag, curr_tag)] += 1
            tag_counts[prev_tag] += 1

    # Convert flat dict to nested dict
    transition_probs: Dict[str, Dict[str, float]] = defaultdict(dict)

    for (prev_tag, curr_tag), count in transition_counts.items():
        transition_probs[prev_tag][curr_tag] = count / tag_counts[prev_tag]

    transition_dict = dict(transition_probs)
    # tolerance = 1e-6  # To handle floating point errors

    # for state, transitions in transition_dict.items():
    #     total = sum(transitions.values())
    #     if abs(total - 1.0) > tolerance:
    #         print(
    #             f"❌ Transition probabilities for '{state}' sum to {total:.6f} (should be 1)"
    #         )
    #     else:
    #         print(f"✅ '{state}' OK (sum = {total:.6f})")
    return transition_dict
