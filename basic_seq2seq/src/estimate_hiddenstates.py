from helpers import model_repo
import sys

BASE_DIR = "C:/Users/Nicolas/Desktop/"


def save_hiddenstate_and_state_vector(prediction, output_file, state_file, sentence):
    with(open(output_file, 'a')) as out_file:
        out_file.write(sentence + ' ')
        for element in prediction[0][0]:
            out_file.write(str(element) + ' ')
        out_file.write('\n')
    #with(open(state_file, 'a')) as out_file:
    #    out_file.write(sentence + ' ')
    #    for element in prediction[1][0]:
    #        out_file.write(str(element) + ' ')
    #    out_file.write('\n')

def batch_predict(in_file):
    # Todo implement this
    raise NotImplementedError("Batch prediction is currently not implemented")
    save_hiddenstate_and_state_vector(hiddenstate, BASE_DIR + "hidden_states_" + suffix + ".txt",
                                      BASE_DIR + "/state_vectors_" + suffix + '.txt',
                                      sentence.replace(' ', '_'))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model = model_repo.argument_model_selection(sys.argv[1])
        in_file = sys.argv[2]
        batch_predict(in_file)
    else:
        model = model_repo.interactive_model_selection()

    mode = input("sent or batch\n")
    suffix = input("File suffix:")
    if mode == '0':
        while True:
            sentence = input("Which sentence should be used?")
            if sentence == '\q':
                exit()
            hiddenstate = model.calculate_hiddenstate_after_encoder(sentence)
            save_hiddenstate_and_state_vector(hiddenstate, BASE_DIR + "hidden_states_" + suffix + ".txt",
                                              BASE_DIR + "/state_vectors_" + suffix + '.txt',
                                              sentence.replace(' ', '_'))
    elif mode == '1':
        in_file = input("source file\n")
        batch_predict(in_file)
    else:
        exit("Only mode 0 and 1 are available")
