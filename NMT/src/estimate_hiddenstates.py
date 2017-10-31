import sys

from helpers import model_repo

BASE_DIR = "C:/Users/Nicolas/Desktop/"


def save_hiddenstates(hiddenstates, output_file, sentences):
    with(open(output_file, 'a')) as out_file:
        for i in range(len(sentences)):
            out_file.write(sentence[i] + ' ')
            for element in hiddenstates[i][0]:
                out_file.write(str(element) + ' ')
            out_file.write('\n')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        model = model_repo.argument_model_selection(sys.argv[1])
        in_file = sys.argv[2]
        split_flag = sys.argv[3]
        suffix = sys.argv[4]

        source_sentences = open(in_file, encoding='UTF-8').read().split('\n')
        if split_flag in ['y', 'yes']:
            temp_source_sentences = source_sentences.copy()
            source_sentences = []
            for line in temp_source_sentences:
                input_text, _ = line.split('\t')
                source_sentences.append(input_text)

        hiddenstates = model.calculate_hiddenstate_after_encoder(source_sentences)
        save_hiddenstates(hiddenstates, BASE_DIR + "hidden_states_" + suffix + ".txt",
                          [sent.replace(' ', '_') for sent in source_sentences])

    else:
        model = model_repo.interactive_model_selection()

    mode = input("sent or batch\n")
    suffix = input("File suffix:")
    if mode == '0':
        while True:
            sentence = input("Which sentence should be used?")
            if sentence == '\q':
                exit()
            sentence = [sentence]
            hiddenstates = model.calculate_hiddenstate_after_encoder(sentence)
            save_hiddenstates(hiddenstates, BASE_DIR + "hidden_states_" + suffix + ".txt",
                              [sent.replace(' ', '_') for sent in source_sentences])
    elif mode == '1':
        in_file = input("source file\n")
        # batch_predict(in_file)
    else:
        exit("Only mode 0 and 1 are available")
