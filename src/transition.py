from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def compute_transition_parameters(
    training_data: List[List[Tuple[str, str]]], tags: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Computes transition probabilities q(y_i | y_{i-1}) with add-one smoothing,
    ensuring only valid transitions are included (e.g., START → tag, tag → STOP, tag → tag).

    Args:
        training_data: list of sentences, each as a list of (word, tag) pairs
        tags: list of all possible tags (excluding START/STOP)

    Returns:
        Nested dictionary of transition probabilities: {prev_tag: {curr_tag: probability}}
    """
    transition_counts = Counter()
    tag_counts = Counter()

    # Count transitions and tag occurrences
    for sentence in training_data:
        tag_sequence = ["START"] + [tag for _, tag in sentence] + ["STOP"]
        for i in range(1, len(tag_sequence)):
            prev_tag = tag_sequence[i - 1]
            curr_tag = tag_sequence[i]
            transition_counts[(prev_tag, curr_tag)] += 1
            tag_counts[prev_tag] += 1

    transition_probs: Dict[str, Dict[str, float]] = defaultdict(dict)

    # Valid transitions:
    # START → tag
    for curr_tag in tags:
        count = transition_counts.get(("START", curr_tag), 0)
        total_prev = tag_counts.get("START", 0)
        transition_probs["START"][curr_tag] = (count + 1) / (total_prev + len(tags))

    # tag → tag or tag → STOP
    for prev_tag in tags:
        possible_next = tags + ["STOP"]
        total_prev = tag_counts.get(prev_tag, 0)
        for curr_tag in possible_next:
            count = transition_counts.get((prev_tag, curr_tag), 0)
            transition_probs[prev_tag][curr_tag] = (count + 1) / (
                total_prev + len(possible_next)
            )

    return dict(transition_probs)
