from helpers import model_repo
import os, sys

"""
Interactive inference of pretrained models via terminal.
"""


# C:/Users/Nicolas/Desktop/own_validation_data.en
# C:/Users/Nicolas/Desktop/deu_val_data.en

def predict_interactive_per_sentence(inference_model):
    """
    Asks the user for a source sentence and translates them with the given model.
    :param inference_model: the model which should be used for prediction
    """
    while True:
        print("\n\nPlease type in the sentence which should be translated:")
        source_sentence = input()
        if source_sentence == '\q':
            exit()
        if source_sentence == '\m':
            inference_model = model_repo.interactive_model_selection()
            continue
        target_sentence = inference_model.predict_one_sentence(source_sentence)

        print("Source sentence:\n", source_sentence)
        print("Translated sentence:\n", target_sentence)


def predict_interactive_from_file(model):
    source_file = input('Path of source file')
    if os.path.exists(source_file) is False:
        exit("source file does not exists")
    out_file_name = input('suffix of output file')
    out_file = os.path.join(os.path.abspath(os.path.join(source_file, os.pardir)), out_file_name + ".pred")

    source_sentences = open(source_file, encoding='UTF-8').read().split('\n')
    predictions = model.predict_batch(source_sentences)
    with(open(out_file, 'w')) as file:
        for sent in predictions:
            file.write(sent + '\n')


if len(sys.argv) > 1:
    model = model_repo.argument_model_selection(sys.argv[1])
    model.predict_batch(sys.argv[2])
else:
    model = model_repo.interactive_model_selection()
    mode = -1
    while mode == -1:
        print("Model", model.identifier, "selected")
        print("Mode: \n0: per sentence prediction\n1: batch prediction")
        mode = input()

    if mode == '0':
        predict_interactive_per_sentence(model)
    elif mode == '1':
        predict_interactive_from_file(model)
        # implement batch prediction
