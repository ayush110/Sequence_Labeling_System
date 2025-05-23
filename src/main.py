import sys
import os

from emission import compute_emission_parameters, emission_predict
from transition import compute_transition_parameters
from viterbi import viterbi_predict
from viterbi_top_k import viterbi_top_k_predict
from structured_perceptron import (
    structured_perceptron_predict,
    train_structured_perceptron
)
from utils import (
    get_sequences_dataset,
    parse_test_data,
    get_unique_states,
    generate_output,
)


def main():

    input_filename = sys.argv[1] if len(sys.argv) > 1 else "dev.in"
    input_path = os.path.join("./EN", input_filename)
    output_prefix = input_filename.replace(".in", "")

    training_data = get_sequences_dataset("./EN/train")
    test_data = parse_test_data(input_path)

    """ PART 1 --- Compute emission parameters """
    emission_parameters = compute_emission_parameters(training_data)
    generate_output(
        test_data,
        lambda sent: emission_predict(sent, emission_parameters),
        "./EN/dev.p1.out",
    )

    """ PART 2 --- Compute transition parameters and run Viterbi """
    # Compute transition parameters using training data
    # count number of unique tags in training data
    tags = list(get_unique_states(training_data))
    transition_parameters = compute_transition_parameters(training_data, tags)
    generate_output(
        test_data,
        lambda sent: viterbi_predict(sent, transition_parameters, emission_parameters),
        "./EN/dev.p2.out",
    )

    """ PART 3 --- 4th Best Viterbi Output Sequence """
    generate_output(
        test_data,
        lambda sent: viterbi_top_k_predict(
            sent, transition_parameters, emission_parameters, top_k=4
        ),
        "./EN/dev.p3.out",
    )

    """ PART 4 --- Structured Perceptron """
    tags = list(set(tag for sentence in training_data for _, tag in sentence))
    trained_weights = train_structured_perceptron(training_data, tags)

    generate_output(
        test_data,
        lambda sent: structured_perceptron_predict(
            sent, training_data, trained_weights, tags
        ),
        f"./EN/{output_prefix}.p4.out",
    )


if __name__ == "__main__":
    main()
