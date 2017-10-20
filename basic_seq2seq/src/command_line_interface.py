"""
Interactive inference of pretrained models via terminal.
"""


def get_inference_module():
    print("Which model do you want to use?")
    for i in range(len(model_names)):
        print(i, "-", model_names[i])
    model_id = int(input())
    if model_id > len(model_names) - 1 or model_id < 0:
        print("error")
        return get_inference_module()
    if model_id == 0:
        from models.char_seq2seq_first_approach import Seq2Seq1
        char_seq2seq1 = Seq2Seq1()
        char_seq2seq1.setup_inference()
        return char_seq2seq1
    elif model_id == 1:
        from models.char_seq2seq_second_approach import Seq2Seq2
        char_seq2seq2 = Seq2Seq2()
        char_seq2seq2.setup_inference()
        return char_seq2seq2
    elif model_id == 2:
        from models.keras_tut_original import KerasTutSeq2Seq
        keras_tut_seq2seq = KerasTutSeq2Seq()
        keras_tut_seq2seq.setup_inference()
        return keras_tut_seq2seq


model_names = ["char_seq2seq_1", "char_seq2seq_2", "char_seq2seq_3"]
model = None

inference_model = get_inference_module()

while True:
    print("\n\nPlease type in the sentence which should be translated:")
    source_sentence = input()
    if source_sentence == '\q':
        exit()
    if source_sentence == '\m':
        inference_model = get_inference_module()
        continue
    target_sentence = inference_model.predict(source_sentence)

    print("Source sentence:\n", source_sentence)
    print("Translated sentence:\n", target_sentence)
