from emission import compute_emission_parameters, generate_output, generate_tags
from transition import compute_transition_parameters
from viterbi import viterbi
from utils import get_sequences_dataset, parse_test_data

def main():

    training_data = get_sequences_dataset("../EN/train")
    test_data = parse_test_data("../EN/dev.in")

    ### PART 1 --- Compute emission parameters ###
    emission_parameters = compute_emission_parameters(training_data)
    tags = generate_tags(emission_parameters)
    generate_output(tags, test_data)

    ### PART 2 Compute transition parameters and run Viterbi ###

    # Compute transition parameters using training data
    transition_parameters = compute_transition_parameters(training_data)

    # Get the full list of tags (states)
    tags = list(emission_parameters.keys())

    # Train the Viterbi algorithm on the training data "train"
    

    # Run the Viterbi algorithm on the test data dev.in

if __name__ == "__main__":
    main()
