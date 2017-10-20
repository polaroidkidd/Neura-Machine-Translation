import numpy as np
from helpers.chkpt_to_three_models import Chkpt_to_Models


# chkpt_to_models = Chkpt_to_Models()
# chkpt_to_models.start()



def estimate_hidden_states(model, sentences, model_identifier):
    model.setup_inference()
    new_h = np.array(model.get_hidden_state(sentences[0]))
    expected_shape = (new_h.shape[0], new_h.shape[2])
    hiddenstates = np.zeros((new_h.shape[0], 2, new_h.shape[2]))
    reshaped_new_h = new_h.reshape(expected_shape)
    hiddenstates[0, :] = reshaped_new_h
    print(reshaped_new_h[0])
    print(reshaped_new_h[1])
    print(reshaped_new_h)
    print("next")

    for idx in range(len(sentences) - 1):
        new_h = np.array(model.get_hidden_state(sentences[idx+1]))
        reshaped_new_h = new_h.reshape(expected_shape)
        hiddenstates[idx + 1, :] = reshaped_new_h
        print(reshaped_new_h[0])
        print(reshaped_new_h[1])
        print(reshaped_new_h)
    print(hiddenstates.shape)
    out_file = model_identifier + '_hiddenstates'
    np.save(out_file, hiddenstates)
    exit()



def start():
    sentences_file = '../../DataSets/sentence2hiddenstate.txt'
    print(sentences_file)
    sentences = open(sentences_file, encoding='UTF-8').read().split('\n')
    print(sentences)

    # Run char_seq2seq_first_approach
    from models.char_seq2seq_first_approach import Seq2Seq1
    estimate_hidden_states(Seq2Seq1(), sentences, 'first_approac')

    # Run char_seq2seq_first_approach
    from models.char_seq2seq_second_approach import Seq2Seq2
    estimate_hidden_states(Seq2Seq2(), sentences,  'second_approach')

    from models.keras_tut_original import KerasTutSeq2Seq
    estimate_hidden_states(KerasTutSeq2Seq(), sentences, 'orig')


#start()
hiddenstates = np.load('first_approac_hiddenstates.npy')
for sentence in hiddenstates:
    print(sentence.shape)
    print(sentence[0])
    print(sentence[1])

# todo what is [0] and what [1]

#plot both in their own plot