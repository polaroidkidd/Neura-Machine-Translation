"""
Interactive inference of pretrained models via terminal.
"""
from inference.char_inference import CharBasedInference


def get_inference_module():
    print("Which model do you want to use?")
    for i in range(len(model_names)):
        print(i, "-", model_names[i][0])
    model_id = int(input())
    if model_id > len(model_names) - 1 or model_id < 0:
        print("error")
        return get_inference_module()
        # TODO get correct inference class
    if model_id == 0:
        return CharBasedInference("model_file", "input_token_idx_file", "output_token_idx_file")


model_names = [["charbased", "path"], ["wordbased", "path"]]
model = None

inference_model = get_inference_module()

print("Please type in the sentence which should be translated:")
source_sentence = input()

target_sentence = inference_model.translate(model, source_sentence)

print("Source sentence:\n", source_sentence)
print("Translated sentence:\n", target_sentence)
