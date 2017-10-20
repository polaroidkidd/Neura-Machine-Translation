import numpy as np
from helpers.chkpt_to_three_models import Chkpt_to_Models

#chkpt_to_models = Chkpt_to_Models()
#chkpt_to_models.start()


# Run char_seq2seq_first_approach
from models.char_seq2seq_first_approach import Seq2Seq1
char_seq2seq1 = Seq2Seq1()
#char_seq2seq1.start_training()
char_seq2seq1.setup_inference()

print(char_seq2seq1.get_hidden_state("hello mr president"))
exit()
#char_seq2seq1.predict('Hello Mr President')

print('\n\n\n')

# Run char_seq2seq_first_approach
from models.char_seq2seq_second_approach import Seq2Seq2
char_seq2seq2 = Seq2Seq2()
#char_seq2seq2.start_training()
#char_seq2seq2.setup_inference()
print(char_seq2seq2.get_hidden_state("hello mr president"))
#char_seq2seq2.predict('Hello Mr President')


from models.keras_tut_original import KerasTutSeq2Seq
keras_tut_seq2seq = KerasTutSeq2Seq()
#keras_tut_seq2seq.start_training()
keras_tut_seq2seq.setup_inference()
print(keras_tut_seq2seq.get_hidden_state("hello mr president"))
#keras_tut_seq2seq.predict('Hello Mr President')

np.savetxt()
