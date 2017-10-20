from helpers import model_repo

"""
Interactive inference of pretrained models via terminal.
"""
import sys


def predict_interactive_per_sentence():
    while True:
        print("\n\nPlease type in the sentence which should be translated:")
        source_sentence = input()
        if source_sentence == '\q':
            exit()
        if source_sentence == '\m':
            inference_model = model_repo.interactive_model_selection()
            continue
        target_sentence = inference_model.predict(source_sentence)

        print("Source sentence:\n", source_sentence)
        print("Translated sentence:\n", target_sentence)


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

    if mode == 0:
        predict_interactive_per_sentence()
    elif mode == 1:
        pass
        # implement batch prediction
