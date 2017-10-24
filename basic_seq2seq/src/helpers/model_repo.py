from models import CharSeq2SeqTutOneHotInput, CharSeq2SeqTutIndexInput
from models import WordSeq2SeqTutNoStartEndToken
from models import WordSeq2SeqTutStartEndTokenNoUNK
from models import WordSeq2SeqTutStartEndUnkToken
from models import char_seq2seq_second_approach
from models import model_1
from models import model_1_embedding2
from models import model_2
from models import model_2_token_also_at_encoder
from models import model_2_token_also_at_encoder_unk
from models import model_2_unk
from models import model_2_without_dropout

# from models import model_2_token_also_at_encoder_without_dropout
# from models import model_2_token_also_at_encoder_without_dropout_unk
# from models import model_2_unk_without_dropout


models = {'char_seq2seq_second_approach': '0',
          'CharSeq2SeqTutIndexInput': '1',
          'CharSeq2SeqTutOneHotInput': '2',
          'model_1': '3',
          'model_1_embedding2': '4',
          'model_2': '5',
          'model_2_without_dropout': '6',
          'WordSeq2SeqTutNoStartEndToken': '7',
          'WordSeq2SeqTutStartEndTokenNoUNK': '8',
          'model_2_token_also_at_encoder': '9',
          'model_2_token_also_at_encoder_unk': '10',
          'model_2_unk': '11',
          'WordSeq2SeqTutStartEndUnkToken': '12'
          }


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

    elif searched_model == 'model_2_without_dropout' or searched_model == models['model_2_without_dropout']:
        model = model_2_without_dropout.Seq2Seq2()

    elif searched_model == 'WordSeq2SeqTutNoStartEndToken' or searched_model == models['WordSeq2SeqTutNoStartEndToken']:
        model = WordSeq2SeqTutNoStartEndToken.Seq2Seq2()

    elif searched_model == 'WordSeq2SeqTutStartEndTokenNoUNK' or searched_model == models[
        'WordSeq2SeqTutStartEndTokenNoUNK']:
        model = WordSeq2SeqTutStartEndTokenNoUNK.Seq2Seq2()

    elif searched_model == 'model_2_token_also_at_encoder' or searched_model == models['model_2_token_also_at_encoder']:
        model = model_2_token_also_at_encoder.Seq2Seq2()

    elif searched_model == 'model_2_token_also_at_encoder_unk' or searched_model == models[
        'model_2_token_also_at_encoder_unk']:
        model = model_2_token_also_at_encoder_unk.Seq2Seq2()

    elif searched_model == 'model_2_unk' or searched_model == models['model_2_unk']:
        model = model_2_unk.Seq2Seq2()

    elif searched_model == 'WordSeq2SeqTutStartEndUnkToken' or searched_model == models['WordSeq2SeqTutStartEndUnkToken']:
        model = WordSeq2SeqTutStartEndUnkToken.Seq2Seq2()

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
