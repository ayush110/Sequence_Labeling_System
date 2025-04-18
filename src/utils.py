from typing import List, Tuple, Set

iomport numpy as np

def get_sequences_dataset(dataset_path: str) -> List[List[Tuple[str, str]]]:
    """
    Parses the dataset file into a list of sequences.
    Each sequence is a list of (observation, state) tuples.
    """
    with open(dataset_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    sequences: List[List[Tuple[str, str]]] = []
    seq: List[Tuple[str, str]] = []

    for line in lines:
        if line:
            parts = line.split()
            if len(parts) == 2:
                seq.append((parts[0], parts[1]))
        else:
            if seq:
                sequences.append(seq)
                seq = []
    if seq:
        sequences.append(seq)

    return sequences


def get_unique_states(dataset: List[List[Tuple[str, str]]]) -> Set[str]:
    """
    Extracts the unique states (tags) from the dataset.
    """
    states: Set[str] = set()
    for sequence in dataset:
        for _, state in sequence:
            states.add(state)
    return states


def get_unique_observations(dataset: List[List[Tuple[str, str]]]) -> Set[str]:
    """
    Extracts the unique observations (words) from the dataset.
    """
    observations: Set[str] = set()
    for sequence in dataset:
        for observation, _ in sequence:
            observations.add(observation)
    return observations


def write_predictions_to_file(predictions, output_file_path):
    """
    Example format of predictions dictionary
    predictions = {
        0: [[("Municipal", "B-NP"), ("bonds", "I-NP"), ("are", "B-VP"), ...]],
        1: [[("He", "B-NP"), ("added", "B-VP"), ("that", "B-SBAR"), ...]],
        ...
    }
    """
    with open(output_file_path, "w", encoding="UTF-8") as f:
        for example in predictions:
            for entity in predictions[example]:
                for word, label in entity:
                    f.write(f"{word} {label}\n")
                f.write("\n")


def parse_test_data(test_data_path: str) -> List[List[str]]:
    """
    Parses the test data file into a list of sequences.
    Each sequence is a list of observations (words).
    """
    with open(test_data_path) as f:
        lines = [line.strip() for line in f]

    sequences: List[List[str]] = []
    seq: List[str] = []

    for line in lines:
        if line:
            parts = line.split()
            if len(parts) == 1:
                seq.append(parts[0])
        else:
            if seq:
                sequences.append(seq)
                seq = []
    if seq:
        sequences.append(seq)

    return sequences


def _safe_log(x):
    return -np.inf if x <= 0 else np.log(x)
