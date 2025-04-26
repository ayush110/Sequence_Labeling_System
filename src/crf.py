import math
import os

# ------------------ Utilities ------------------

def read_labeled_data(filepath):
    sentences, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
            else:
                parts = line.split()
                words.append(parts[0])
                tags.append(parts[1])
        if words:
            sentences.append(words)
            labels.append(tags)
    return sentences, labels

def read_unlabeled_data(filepath):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        words = []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    words = []
            else:
                words.append(line)
        if words:
            sentences.append(words)
    return sentences

def write_output(filepath, sentences, tags):
    with open(filepath, 'w', encoding='utf-8') as f:
        for words, labels in zip(sentences, tags):
            for w, l in zip(words, labels):
                f.write(f"{w} {l}\n")
            f.write("\n")

# ------------------ Feature Extraction ------------------

def get_features(words, prev_tag, tag, i):
    feats = []
    feats.append(f"tag={tag}")
    feats.append(f"word={words[i]} tag={tag}")
    feats.append(f"prev_tag={prev_tag} tag={tag}")
    return feats

def compute_feature_counts(words, tags):
    counts = {}
    for i in range(len(words)):
        prev = tags[i - 1] if i > 0 else "<START>"
        for feat in get_features(words, prev, tags[i], i):
            counts[feat] = counts.get(feat, 0.0) + 1.0
    return counts

# ------------------ Forward-Backward ------------------

def compute_log_sum_exp(scores):
    max_score = max(scores)
    sum_exp = sum(math.exp(s - max_score) for s in scores)
    return max_score + math.log(sum_exp)

def compute_partition_function(words, weights, tagset):
    n = len(words)
    alpha = [{} for _ in range(n)]
    for tag in tagset:
        score = sum(weights.get(f, 0.0) for f in get_features(words, "<START>", tag, 0))
        alpha[0][tag] = score

    for i in range(1, n):
        for curr in tagset:
            scores = []
            for prev in tagset:
                trans_score = sum(weights.get(f, 0.0) for f in get_features(words, prev, curr, i))
                scores.append(alpha[i-1][prev] + trans_score)
            alpha[i][curr] = compute_log_sum_exp(scores)

    return compute_log_sum_exp([alpha[-1][t] for t in tagset])

# ------------------ Expected Feature Counts ------------------

def expected_feature_counts(words, weights, tagset):
    n = len(words)
    Z = compute_partition_function(words, weights, tagset)

    alpha = [{} for _ in range(n)]
    beta = [{} for _ in range(n)]

    # Forward
    for tag in tagset:
        alpha[0][tag] = sum(weights.get(f, 0.0) for f in get_features(words, "<START>", tag, 0))

    for i in range(1, n):
        for tag in tagset:
            scores = []
            for prev in tagset:
                score = alpha[i-1][prev] + sum(weights.get(f, 0.0) for f in get_features(words, prev, tag, i))
                scores.append(score)
            alpha[i][tag] = compute_log_sum_exp(scores)

    # Backward
    for tag in tagset:
        beta[n-1][tag] = 0.0

    for i in range(n-2, -1, -1):
        for tag in tagset:
            scores = []
            for next_tag in tagset:
                score = beta[i+1][next_tag] + sum(weights.get(f, 0.0) for f in get_features(words, tag, next_tag, i+1))
                scores.append(score)
            beta[i][tag] = compute_log_sum_exp(scores)

    # Expected counts
    expected = {}
    for i in range(n):
        for prev in tagset if i > 0 else ["<START>"]:
            for curr in tagset:
                score = 0.0
                if i == 0:
                    score = alpha[0][curr] + beta[0][curr]
                else:
                    f = get_features(words, prev, curr, i)
                    total = sum(weights.get(x, 0.0) for x in f)
                    score = alpha[i-1][prev] + total + beta[i][curr]
                prob = math.exp(score - Z)
                for feat in get_features(words, prev, curr, i):
                    expected[feat] = expected.get(feat, 0.0) + prob
    return expected

# ------------------ Viterbi Decoding ------------------

def viterbi(words, weights, tagset):
    n = len(words)
    dp = [{} for _ in range(n)]
    back = [{} for _ in range(n)]

    for tag in tagset:
        score = sum(weights.get(f, 0.0) for f in get_features(words, "<START>", tag, 0))
        dp[0][tag] = score
        back[0][tag] = "<START>"

    for i in range(1, n):
        for curr in tagset:
            best_score = -float('inf')
            best_prev = None
            for prev in tagset:
                score = dp[i-1][prev] + sum(weights.get(f, 0.0) for f in get_features(words, prev, curr, i))
                if score > best_score:
                    best_score = score
                    best_prev = prev
            dp[i][curr] = best_score
            back[i][curr] = best_prev

    last_tag = max(dp[-1], key=dp[-1].get)
    tags = [last_tag]
    for i in range(n-1, 0, -1):
        tags.insert(0, back[i][tags[0]])
    return tags

# ------------------ CRF Training ------------------

def train_crf(train_file, dev_in_file, dev_out_file, output_file, epochs=5, lr=0.1):
    print("Reading data...")
    train_sents, train_tags = read_labeled_data(train_file)
    dev_sents = read_unlabeled_data(dev_in_file)
    tagset = set(tag for tags in train_tags for tag in tags)
    weights = {}

    print("Training CRF...")
    for epoch in range(epochs):
        for x, y in zip(train_sents, train_tags):
            gold_feats = compute_feature_counts(x, y)
            expected_feats = expected_feature_counts(x, weights, tagset)

            for feat, count in gold_feats.items():
                weights[feat] = weights.get(feat, 0.0) + lr * count
            for feat, count in expected_feats.items():
                weights[feat] = weights.get(feat, 0.0) - lr * count
        print(f"Epoch {epoch+1} done.")

    print("Predicting on dev set...")
    predictions = [viterbi(sent, weights, tagset) for sent in dev_sents]
    write_output(output_file, dev_sents, predictions)
    print(f"Output written to {output_file}")

# ------------------ Main ------------------

if __name__ == "__main__":
    train_file = "EN/train"
    dev_in_file = "EN/dev.in"
    dev_out_file = "EN/dev.out"
    output_file = "EN/dev.p44.out"
    train_crf(train_file, dev_in_file, dev_out_file, output_file, epochs=5, lr=0.1)
