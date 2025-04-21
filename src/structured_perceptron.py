from collections import defaultdict, Counter
from typing import List, Tuple, Dict

def preprocess_data(training_data: List[List[Tuple[str, str]]], k: int = 3) -> List[List[Tuple[str, str]]]:
    word_counts = Counter(word for sentence in training_data for word, _ in sentence)
    return [
        [(word if word_counts[word] >= k else "#UNK#", tag) for word, tag in sentence]
        for sentence in training_data
    ]

def init_weights(training_data: List[List[Tuple[str, str]]], tags: List[str]) -> Dict:
    """
    Initializes weights for the structured perceptron.
    
    Args:
        training_data (List[List[Tuple[str, str]]]): List of sentences, where each sentence is a list of (word, tag) pairs.
        tags (List[str]): List of all possible tags.
    """
    weights = defaultdict(float)

    for tag in tags:
        for word in set(word for sentence in training_data for word, _ in sentence):
            weights[(tag, word)] = 0.0
            weights[(tag, "CAPITALIZED")] = 0.0
            weights[(tag, "IS_NUMBER")] = 0.0

            if len(word) > 3:
                weights[(tag, f"SUFFIX_{word[-3:]}")] = 0.0
            if len(word) > 2:
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
    features = defaultdict(float)

    for i, word in enumerate(sentence):
        tag = tags[i]
        prev_tag = tags[i - 1] if i > 0 else "START"

        # emission feature: (tag, word)
        features[(tag, word)] += 1.0
        # transition feature: (prev_tag, tag)
        features[(prev_tag, tag)] += 1.0

        if word == "#UNK#":
           features[(tag, "UNK")] += 1.0  # unknown word feature
        if word[0].isupper():
            features[(tag, "CAPITALIZED")] += 1.0  # capitalization feature
        if word.isdigit() or (word.replace('.', '', 1).isdigit() and word.count('.') == 1):
            features[(tag, "IS_NUMBER")] += 1.0  # numeric feature
        
        # predefine some common suffixes and prefixes
        # and add features for them
        common_4_letter_suffixes = {"tion", "ment", "ness", "sion", "able", "ible", "ship", "hood"}
        common_3_letter_suffixes = {"ing", "est", "ous", "ive", "ity", "ity"}
        common_prefixes = {"un", "re", "in", "en", "de"}
        if len(word) > 4 and word[-4:] in common_4_letter_suffixes:
            features[(tag, f"SUFFIX_{word[-4:]}")] += 1.0
        if len(word) > 3 and word[-3:] in common_3_letter_suffixes:
            features[(tag, f"SUFFIX_{word[-3:]}")] += 1.0
        if len(word) > 2 and word[:2] in common_prefixes:
            features[(tag, f"PREFIX_{word[:2]}")] += 1.0

    # add transition to STOP
    features[(tags[-1], "STOP")] += 1.0
    return features

def extract_local_features(word: str, curr_tag: str, prev_tag: str, is_last_word: bool) -> Dict[Tuple, int]:
    """
    Extracts local features for a given word, its tag, and the previous tag.

    Args:
        word (str): The current word.
        curr_tag (str): The current tag.
        prev_tag (str): The previous tag.
        is_last_word (bool): Whether the current word is the last word in the sentence.
    """
    features = defaultdict(float)

    # emission feature: (tag, word)
    features[(curr_tag, word)] += 1.0

    # transition feature: (prev_tag, curr_tag)
    features[(prev_tag, curr_tag)] += 1.0

    if word == "#UNK#":
        features[(curr_tag, "UNK")] += 1.0
    if word[0].isupper():
        features[(curr_tag, "CAPITALIZED")] += 1.0
    if word.isdigit() or (word.replace('.', '', 1).isdigit() and word.count('.') == 1):
        features[(curr_tag, "IS_NUMBER")] += 1.0  # numeric feature

    # predefine some common suffixes and prefixes
    # and add features for them
    common_4_letter_suffixes = {"tion", "ment", "ness", "sion", "able", "ible", "ship", "hood"}
    common_3_letter_suffixes = {"ing", "est", "ous", "ive", "ity", "ity"}
    common_prefixes = {"un", "re", "in", "en", "de"}
    if len(word) > 4 and word[-4:] in common_4_letter_suffixes:
        features[(curr_tag, f"SUFFIX_{word[-4:]}")] += 1.0
    if len(word) > 3 and word[-3:] in common_3_letter_suffixes:
        features[(curr_tag, f"SUFFIX_{word[-3:]}")] += 1.0
    if len(word) > 2 and word[:2] in common_prefixes:
        features[(curr_tag, f"PREFIX_{word[:2]}")] += 1.0
    
    if is_last_word:
        features[(curr_tag, "STOP")] += 1.0
    
    return features

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
    backpointer[0]["START"] = None

    # forward pass
    for i in range(1, n + 1):
        word = sentence[i - 1]
        for curr_tag in tags:
            max_score, best_prev_tag = float("-inf"), None
            for prev_tag in dp[i - 1]:
                local_features = extract_local_features(word, curr_tag, prev_tag, i == n)
                local_score = sum(weights[feature] * count for feature, count in local_features.items())

                score = dp[i - 1][prev_tag] + local_score

                if score > max_score:
                    max_score = score
                    best_prev_tag = prev_tag

            dp[i][curr_tag] = max_score
            backpointer[i][curr_tag] = best_prev_tag

    # transition to STOP
    max_score, best_last_tag = float("-inf"), None
    for prev_tag in dp[n]:
        local_features = extract_local_features("STOP", prev_tag, "STOP", True)
        local_score = sum(weights[feature] * count for feature, count in local_features.items())

        score = dp[n][prev_tag] + local_score
        if score > max_score:
            max_score = score
            best_last_tag = prev_tag

    # backward pass to reconstruct the best sequence
    tags = []
    current_tag = best_last_tag
    for i in range(n, 0, -1):
        tags.insert(0, current_tag)
        current_tag = backpointer[i][current_tag]

    return tags

def train_structured_perceptron(training_data: List[List[Tuple[str, str]]], tags: List[str], max_iter = 2) -> Dict:
    """
    Trains the Structured Perceptron on the given training data.

    Args:
        training_data (List[List[Tuple[str, str]]]): List of sentences, where each sentence is a list of (word, tag) pairs.
    """
    # initialize weights
    training_data = preprocess_data(training_data)
    weights = init_weights(training_data, tags)

    # iterate for some provided # of iterations
    for iteration in range(max_iter):
        print(f"currently on iteration {iteration + 1}/{max_iter}")
        # go through each sentence in the training data    
        for index, sentence in enumerate(training_data):
            print(f"currently on sentence {index + 1}/{len(training_data)}")
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
