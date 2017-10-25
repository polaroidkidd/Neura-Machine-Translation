from models import WordBasedSeq2Seq
from models import model_2_token_also_at_encoder_unk

models = {'WordBasedSeq2Seq': '0',
          'model_2_token_also_at_encoder_unk': '1'
          }


def determine_model(searched_model):
    model = -1
    if searched_model == 'WordBasedSeq2Seq' or searched_model == models['WordBasedSeq2Seq']:
        model = WordBasedSeq2Seq.Seq2Seq2()

    elif searched_model == 'model_2_token_also_at_encoder_unk' or searched_model == models[
        'model_2_token_also_at_encoder_unk']:
        model = model_2_token_also_at_encoder_unk.Seq2Seq2()

    return model


def print_models():
    for model in models:
        print("-", model, ":", models[model])


def interactive_model_selection():
    print("Which model do you want to use?")

    model = -1
    while model == -1:
        print_models()
        choosed_model_code = input()
        if choosed_model_code == '\q':
            exit(0)

        model = determine_model(choosed_model_code)
        if model == -1:
            print("\nThis model doesn't exists. Following models are allowed:")
    return model


def argument_model_selection(argument):
    model = determine_model(argument)
    if model == -1:
        exit("Error: Model does not exists")
    return model
