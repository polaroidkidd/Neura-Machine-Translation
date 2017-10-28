import re
from collections import OrderedDict

from nltk import TreebankWordTokenizer


class Tokenizer():
    def __init__(self, unk_token, start_token, end_token, num_words=None):
        self.treebank_word_tokenizer = TreebankWordTokenizer()
        improved_open_quote_regex = re.compile(u'([«“‘])', re.U)
        improved_close_quote_regex = re.compile(u'([»”’])', re.U)
        improved_punct_regex = re.compile(r'([^\.])(\.)([\]\)}>"\'' u'»”’ ' r']*)\s*$', re.U)
        self.treebank_word_tokenizer.STARTING_QUOTES.insert(0, (improved_open_quote_regex, r' \1 '))
        self.treebank_word_tokenizer.ENDING_QUOTES.insert(0, (improved_close_quote_regex, r' \1 '))
        self.treebank_word_tokenizer.PUNCTUATION.insert(0, (improved_punct_regex, r'\1 \2 \3 '))

        self.word_counts = OrderedDict()
        self.word_docs = {}
        self.num_words = num_words
        self.document_count = 0

        self.START_TOKEN = start_token
        self.END_TOKEN = end_token
        self.UNK_TOKEN = unk_token

    def fit_on_texts(self, texts):
        self.document_count = 0
        for text in texts:
            self.document_count += 1
            seq = self.treebank_word_tokenizer.tokenize(text)
            for word in seq:
                if word in self.word_counts:
                    self.word_counts[word] += 1
                else:
                    self.word_counts[word] = 1
            for word in set(seq):
                if word in self.word_docs:
                    self.word_docs[word] += 1
                else:
                    self.word_docs[word] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]

        # note that indices 0,1,2,3 are reserved, never assigned to an existing word
        special_token_count = 3
        self.word_index = dict(
            list(zip(sorted_voc, list(range(1 + special_token_count, len(sorted_voc) + 1 + special_token_count)))))
        self.word_index[self.START_TOKEN] = 1
        self.word_index[self.END_TOKEN] = 2
        self.word_index[self.UNK_TOKEN] = 3
        index_docs = {}
        for word, count in list(self.word_docs.items()):
            index_docs[self.word_index[word]] = count

    def texts_to_sequences(self, texts, search_related_word=False):
        res = []
        for vect in self.__texts_to_sequences_generator(texts, search_related_word):
            res.append(vect)
        return res

    def __texts_to_sequences_generator(self, texts, search_related_word):
        """Transforms each text in texts in a sequence of integers.

        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Yields
            Yields individual sequences.
        """
        num_words = self.num_words
        for text in texts:
            seq = self.treebank_word_tokenizer.tokenize(text)
            seq = [self.START_TOKEN] + seq + [self.END_TOKEN]
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if search_related_word is True:
                            self.__find_related_known_word(w)
                        else:
                            continue
                            # TODO: what is with out of vocab token? (fasttext and glove)
                    else:
                        vect.append(i)
                else:
                    if search_related_word is True:
                        related_word = self.__find_related_known_word(2)
                        i = self.word_index.get(related_word)
                        if i is not None:
                            vect.append(i)
                            continue
                    vect.append(self.word_index.get(self.UNK_TOKEN))
            yield vect
