from src.emission import train_emission_params
from src.baseline_tagger import baseline_tagging
from src.utils import read_data, write_output

# Step 1: Read & preprocess training data
train_data = read_data("EN/train")
smoothed_data = preprocess_rare_words(train_data, k=3)

# Step 2: Train emission parameters
emission_params = train_emission_params(smoothed_data)

# Step 3: Read dev input and tag it
dev_input = read_data("EN/dev.in", is_labeled=False)
predicted_tags = baseline_tagging(dev_input, emission_params)

# Step 4: Write output
write_output(predicted_tags, "outputs/dev.p2.out")
