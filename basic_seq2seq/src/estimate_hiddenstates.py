import numpy as np
from helpers.chkpt_to_three_models import Chkpt_to_Models
from helpers import model_repo
import sys


# TODO implement:
# ability to calculate hiddenstates interactive and via arguments
# to read from file with sentences and write to file sentence and hiddenstates

def save_hiddenstate(hiddenstate, output_file, state_file, sentence):
    print(hiddenstate[0][0].shape)
    print('\n')
    print(hiddenstate[1][0].shape)
    with(open(output_file, 'a')) as out_file:
        out_file.write(sentence + ' ')
        for element in hiddenstate[0][0]:
            out_file.write(str(element) + ' ')
        out_file.write('\n')
    with(open(state_file, 'a')) as out_file:
        out_file.write(sentence + ' ')
        for element in hiddenstate[1][0]:
            out_file.write(str(element) + ' ')
        out_file.write('\n')


if len(sys.argv) > 1:
    model = model_repo.argument_model_selection(sys.argv[1])
    model.predict_batch(sys.argv[2])
else:
    model = model_repo.interactive_model_selection()
    mode = -1

mode = input("sent or batch\n")
suffix = input("File suffix:")
if mode == '0':
    while True:
        sentence = input("Which sentence should be used?")
        if sentence == '\q':
            exit()
        hiddenstate = model.calculate_hiddenstate_after_encoder(sentence)
        save_hiddenstate(hiddenstate, "C:/Users/Nicolas/Desktop/hidden_states.txt",
                         "C:/Users/Nicolas/Desktop/state_vectors_" + suffix + '.txt', sentence.replace(' ', '_'))


elif mode == '1':
    pass
else:
    exit("Only mode 0 and 1 are available")
