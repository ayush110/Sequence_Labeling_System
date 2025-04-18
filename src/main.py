from emission import compute_emission_parameters, generate_output, generate_tags
from transition import compute_transition_parameters
from viterbi import generate_viterbi_output, viterbi
from utils import get_sequences_dataset, parse_test_data

def main():

    training_data = get_sequences_dataset("../EN/train")
    test_data = parse_test_data("../EN/dev.in")

    """ PART 1 --- Compute emission parameters """
    emission_parameters = compute_emission_parameters(training_data)
    tags = generate_tags(emission_parameters)
    #generate_output(tags, test_data)

    """ PART 2 --- Compute transition parameters and run Viterbi """
    # Compute transition parameters using training data
    #count number of unique tags in training data 
    tag_count = len(set(tag for sentence in training_data for _, tag in sentence))
    transition_parameters = compute_transition_parameters(training_data, tag_count)

    # Run the Viterbi algorithm on the test data dev.in
    generate_viterbi_output(test_data, transition_parameters, emission_parameters)

    

    """ PART 3 --- 4th Best Viterbi Output Sequence """

if __name__ == "__main__":
    main()
