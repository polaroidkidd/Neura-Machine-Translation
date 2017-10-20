from models import char_seq2seq_second_approach
from models import CharSeq2SeqTutIndexInput
from models import CharSeq2SeqTutOneHotInput
from models import model_1
from models import model_1_embedding2
from models import model_2
from models import WordSeq2SeqTutNoStartEndToken
from models import WordSeq2SeqTutStartEndTokenNoUNK

models = {'char_seq2seq_second_approach': '0',
          'CharSeq2SeqTutIndexInput': '1',
          'CharSeq2SeqTutOneHotInput': '2',
          'model_1': '3',
          'model_1_embedding2': '4',
          'model_2': '5',
          'WordSeq2SeqTutNoStartEndToken': '6',
          'WordSeq2SeqTutStartEndTokenNoUNK': '7'}


def determine_model(searched_model):
    model = -1
    if searched_model == 'char_seq2seq_second_approach' or searched_model == models['char_seq2seq_second_approach']:
        model = char_seq2seq_second_approach.Seq2Seq2()

    elif searched_model == 'CharSeq2SeqTutIndexInput' or searched_model == models['CharSeq2SeqTutIndexInput']:
        model = CharSeq2SeqTutIndexInput.Seq2Seq2()

    elif searched_model == 'CharSeq2SeqTutOneHotInput' or searched_model == models['CharSeq2SeqTutOneHotInput']:
        model = CharSeq2SeqTutOneHotInput.Seq2Seq2()

    elif searched_model == 'model_1' or searched_model == models['model_1']:
        model = model_1.Seq2Seq2()

    elif searched_model == 'model_1_embedding2' or searched_model == models['model_1_embedding2']:
        model = model_1_embedding2.Seq2Seq2()

    elif searched_model == 'model_2' or searched_model == models['model_2']:
        model = model_2.Seq2Seq2()

    elif searched_model == 'WordSeq2SeqTutNoStartEndToken' or searched_model == models['WordSeq2SeqTutNoStartEndToken']:
        model = WordSeq2SeqTutNoStartEndToken.Seq2Seq2()

    elif searched_model == 'WordSeq2SeqTutStartEndTokenNoUNK' or searched_model == models['WordSeq2SeqTutStartEndTokenNoUNK']:
        model = WordSeq2SeqTutStartEndTokenNoUNK.Seq2Seq2()
    return model

def print_models():
    for model in models:
        print("-", model, ":", models[model])


def interactive_model_selection():
    print("Which model do you want to train?")

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