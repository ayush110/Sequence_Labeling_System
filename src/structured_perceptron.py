from collections import defaultdict
from typing import List, Tuple, Dict

def init_weights(training_data: List[List[Tuple[str, str]]], tags: List[str], words: List[str]) -> Dict:
    weights = defaultdict(float)

    for tag in tags:
        for word in set(word for sentence in training_data for word, _ in sentence):
            weights[(tag, word)] = 0.0
            weights[(tag, "CAPITALIZED")] = 0.0
            weights[(tag, "IS_DIGIT")] = 0.0
            weights[(tag, f"SUFFIX_{word[-3:]}")] = 0.0
            weights[(tag, f"PREFIX_{word[:2]}")] = 0.0
            weights[(tag, "STOP")] = 0.0
            weights[(tag, "#UNK#")] = 0.0
    
    return weights

def extract_features(sentence: List[str], tags: List[str]) -> Dict[Tuple, int]:
    """
    Extracts features for a given sentence and its corresponding tags.

    Args:
        sentence (List[str]): List of words in the sentence.
        tags (List[str]): List of tags corresponding to the words.

    Returns:
        Dict[Tuple, int]: Feature counts for the sentence and tags.
    """
    features = defaultdict(int)
    for i, word in enumerate(sentence):
        tag = tags[i]
        prev_tag = tags[i - 1] if i > 0 else "START"

        # emission feature: (tag, word)
        features[(tag, word)] += 1
        # transition feature: (prev_tag, tag)
        features[(prev_tag, tag)] += 1

        if word == "#UNK#":
           features[(tag, "UNK")] += 1  # unknown word feature
        if word[0].isupper():
            features[(tag, "CAPITALIZED")] += 1  # capitalization feature
        if word.isdigit():
            features[(tag, "IS_DIGIT")] += 1  # numeric feature
        
        # predefine some common suffixes and prefixes
        # and add features for them
        common_suffixes = {"tion", "ment", "ness", "ity", "ing", "ed", "ize"}
        common_prefixes = {"un", "re", "pre", "anti"}
        if len(word) > 3 and word[-3:] in common_suffixes:
            features[(tag, f"SUFFIX_{word[-3:]}")] += 1
        if len(word) > 2 and word[:2] in common_prefixes:
            features[(tag, f"PREFIX_{word[:2]}")] += 1

    # add transition to STOP
    features[(tags[-1], "STOP")] += 1
    return features

def score_sequence(sentence: List[str], weights: Dict, tags: List[str]) -> float:
    """
    Computes the score of a sequence of tags for a given sentence.

    Args:
        sentence (List[str]): List of words in the sentence.
        tags (List[str]): List of tags corresponding to the words.

    Returns:
        float: Score of the sequence.
    """
    features = extract_features(sentence, tags)
    return sum(weights[feature] * count for feature, count in features.items())

def viterbi_decode(sentence: List[str], weights: Dict, tags: List[str]) -> List[str]:
    """
    Decodes the most likely sequence of tags for a given sentence using the Viterbi algorithm.

    Args:
        sentence (List[str]): List of words in the sentence.

    Returns:
        List[str]: Most likely sequence of tags.
    """
    n = len(sentence)
    dp = [{} for _ in range(n + 1)]  # dp table
    backpointer = [{} for _ in range(n + 1)]  # backpointer table

    # initialization
    dp[0]["START"] = 0.0

    # forward pass
    for i in range(1, n + 1):
        word = sentence[i - 1]
        for curr_tag in tags:
            max_score, best_prev_tag = float("-inf"), None
            for prev_tag in dp[i - 1]:
                score = dp[i - 1][prev_tag] + weights[(prev_tag, curr_tag)] + weights[(curr_tag, word)]
                if score > max_score:
                    max_score, best_prev_tag = score, prev_tag
            dp[i][curr_tag] = max_score
            backpointer[i][curr_tag] = best_prev_tag

    # transition to STOP
    max_score, best_last_tag = float("-inf"), None
    for prev_tag in dp[n]:        
        score = dp[n][prev_tag] + weights[(prev_tag, "STOP")]
        if score > max_score:
            max_score, best_last_tag = score, prev_tag

    # backward pass
    tags = [best_last_tag]
    for i in range(n, 1, -1):
        tags.insert(0, backpointer[i][tags[0]])

    return tags

def train_structured_perceptron(training_data: List[List[Tuple[str, str]]], tags: List[str], max_iter = 4) -> Dict:
    """
    Trains the Structured Perceptron on the given training data.

    Args:
        training_data (List[List[Tuple[str, str]]]): List of sentences, where each sentence is a list of (word, tag) pairs.
    """
    # initialize weights
    training_words = set(word for sentence in training_data for word, _ in sentence)
    weights = init_weights(training_data, tags, training_words)

    # iterate for some provided # of iterations
    for iteration in range(max_iter):
        print(f"currently on iteration {iteration + 1}/{max_iter}")

        # go through each sentence in the training data    
        for sentence in training_data:
            # grab words and associated gold tags
            words = [word for word, _ in sentence]
            gold_tags = [tag for _, tag in sentence]

            # decode the predicted tags
            predicted_tags = viterbi_decode(words, weights, tags)

            # update weights if prediction doesn't match gold tags
            if predicted_tags != gold_tags:
                # extract features for both gold and predicted tag sequences
                gold_features = extract_features(words, gold_tags)
                predicted_features = extract_features(words, predicted_tags)

                # update weights (add the gold features and subtract the predicted features)
                for feature, count in gold_features.items():
                    weights[feature] += count
                for feature, count in predicted_features.items():
                    weights[feature] -= count

    return weights

def structured_perceptron_predict(sentence: List[str], training_data: List[List[Tuple[str, str]]], weights: Dict, tags: List[str]) -> List[str]:   
    known_words = set(word for sentence in training_data for word, _ in sentence)
    # replace unknown words with #UNK#
    mapped_sentence = [word if word in known_words else "#UNK#" for word in sentence]
    # predict the tags for the input sentence
    predicted_tags = viterbi_decode(mapped_sentence, weights, tags)

    return predicted_tags
