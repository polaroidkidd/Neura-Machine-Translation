"""
Interactive inference of pretrained models.
"""


def load_pretrained_model(model_file):
    """
    Loads an pretrained model from the given file
    :param model_file:
    :return:
    """
    if model_file is None:
        pass
        # TODO how to throw exception
    print("\nModel is loading...\n")
    return None


def preprocess_source_sentence(source_sentence):
    preprocessed_sentence = None
    return preprocessed_sentence


def prediction_to_sentence(prediction):
    """
    Converts a prediction to a sentence
    :param prediction: 
    :return:
    """
    pass


def translate(model, source_sentence):
    preprocessed_src_sentence = preprocess_source_sentence(source_sentence)
    prediction = model.predict(preprocessed_src_sentence)
    target_sentence = prediction_to_sentence(prediction)
    return target_sentence


def set_model():
    print("Which model do you want to use?")
    for i in range(len(model_names)):
        print(i, "-", model_names[i][0])
    model_id = int(input())
    if model_id > len(model_names) - 1 or model_id < 0:
        print("error")
        set_model()
    else:
        load_pretrained_model(model_names[model_id][1])


model_names = [["charbased", "path"], ["wordbased", "path"]]
model = None

set_model()

print("Please type in the sentence which should be translated:")
source_sentence = input()

target_sentence = translate(model, source_sentence)

print("Source sentence:\n", source_sentence)
print("Translated sentence:\n", target_sentence)
