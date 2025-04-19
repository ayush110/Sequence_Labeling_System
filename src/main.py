from emission import compute_emission_parameters, emission_predict
from transition import compute_transition_parameters
from viterbi import viterbi_predict
from viterbi_top_k import viterbi_top_k_predict
from utils import (
    get_sequences_dataset,
    parse_test_data,
    get_unique_states,
    generate_output,
)


def main():

    training_data = get_sequences_dataset("../EN/train")
    test_data = parse_test_data("../EN/dev.in")

    """ PART 1 --- Compute emission parameters """
    emission_parameters = compute_emission_parameters(training_data)
    generate_output(
        test_data,
        lambda sent: emission_predict(sent, emission_parameters),
        "../EN/dev.p1.out",
    )

    """ PART 2 --- Compute transition parameters and run Viterbi """
    # Compute transition parameters using training data
    # count number of unique tags in training data
    tags = list(get_unique_states(training_data))
    transition_parameters = compute_transition_parameters(training_data, tags)
    generate_output(
        test_data,
        lambda sent: viterbi_predict(sent, transition_parameters, emission_parameters),
        "../EN/dev.p2.out",
    )

    """ PART 3 --- 4th Best Viterbi Output Sequence """
    generate_output(
        test_data,
        lambda sent: viterbi_top_k_predict(
            sent, transition_parameters, emission_parameters, top_k=4
        ),
        "../EN/dev.p3.out",
    )

    """ PART 4 --- 5 """


if __name__ == "__main__":
    main()
