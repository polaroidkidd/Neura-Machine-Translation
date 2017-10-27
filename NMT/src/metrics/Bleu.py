import os
from datetime import datetime

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction

from metrics.BaseMetric import BaseMetric


class Bleu(BaseMetric):
    def __init__(self, model: str, metric: str, timestamp: bool = False):
        """
        This is a wrapper class for nltk's bleu score evaluation designed to work on individual statements
        and reference/hypothesis files.

        :param model: The name of the model which was used to generate the hypothesis
        :param timestamp: if set to true the file name will be appended by a time stamp (for bookkeeping purposes)
        :param metric: The metric used to evaluate the hypothesis

        """
        BaseMetric.__init__(self)
        self.params['model'] = model
        self.params['metric'] = metric
        if timestamp:
            self.params['timestamp'] = datetime.strftime(datetime.now(), "%Y-%m-%d__%H-%M-%S")
        else:
            self.params['timestamp'] = ''
        self.params['hypothesis_reference'] = {
            'hyp': [],
            'ref': []
            }

        self.params['RESULT_DIR'] = '../../../Evaluations/' + self.params['model']
        if not os.path.exists(self.params['RESULT_DIR']):
            os.mkdir(self.params['RESULT_DIR'])
        self.params['FILE_NAME'] = model + '_' + self.params['timestamp'] + '_BLEU.txt'
        self.params['FILE_PATH'] = self.params['RESULT_DIR'] + '/' + self.params['FILE_NAME']

    def evaluate_hypothesis_single(self, hypothesis: str, references: list or str):
        """
        Evaluates predictions via the bleu score metric
        :param references: A list of references against which the predicted string is measured. If only one reference
        is available this method accepts it as a string and converts it to a single-item list
        :param hypothesis: The predicted string(s)
        """

        self.params['hypothesis_reference']['hyp'] = hypothesis.strip('\n').split(' ')
        if '\t' not in references:
            self.params['hypothesis_reference']['ref'] = [references.strip('\n').split(' ')]
        else:
            refs = references.split('\t')
            for i in refs:
                self.params['hypothesis_reference']['ref'].append(i)

        if os.path.exists(self.params['FILE_PATH']):
            with open(self.params['FILE_PATH'], 'a') as file:
                self.__write_single_or_batch_single(file,
                                                    bleu_score.sentence_bleu(
                                                            references=self.params['hypothesis_reference']['ref'],
                                                            hypothesis=self.params['hypothesis_reference']['hyp'],
                                                            smoothing_function=SmoothingFunction().method1,
                                                            auto_reweigh=True),
                                                    self.params['hypothesis_reference']['ref'],
                                                    self.params['hypothesis_reference']['hyp'])
        else:
            with open(self.params['FILE_PATH'], 'w') as file:
                self.__write_single_or_batch_single(file,
                                                    bleu_score.sentence_bleu(
                                                            references=self.params['hypothesis_reference']['ref'],
                                                            hypothesis=self.params['hypothesis_reference']['hyp']),
                                                    self.params['hypothesis_reference']['ref'],
                                                    self.params['hypothesis_reference']['hyp'])

    def evaluate_hypothesis_batch_single(self, hypothesis: str, references: str):
        """
        This method loops through two separate files (references and hypothesis), calculates the BLEU scores of each
        hypothesis and writes it into a file located in the director evaluations/model_name/

        :param references: A text file (including path) containing all the references. If there are multiple for one
        translation the references these should be separated by a '\t'. Each reference should be listed on it's own line
        :param hypothesis: A text file (including path)containing all the hypothesis, one per line
        :return: This method writes the respective BLEU scores into the file specified at class instantiation time.
        """
        if not (os.path.exists(references) or os.path.exists(hypothesis)):
            raise FileNotFoundError
        else:
            with \
                    open(hypothesis, 'r', encoding='utf-8') as hyp_file, \
                    open(references, 'r', encoding='utf-8') as ref_file:
                for hypothesis_line, references_line in zip(hyp_file, ref_file):
                    self.evaluate_hypothesis_single(hypothesis_line.strip('\n'), references_line.strip('\n'))

    def evaluate_hypothesis_corpus(self, hypothesis: str, references: str):
        """
        Wrapper method for evaluating an entire corpus.

        :param hypothesis: A file (including path) containing all hypothesis. Hypothesis are separated by '\n'
        :param references: A file (including path) containing all references. If multiple references exist for one
        translation they have to be separated by a '\t'
        :return: The result is written into the evaluations directory
        """
        if not (os.path.exists(references) or os.path.exists(hypothesis)):
            raise FileNotFoundError
        else:
            with \
                    open(hypothesis, 'r', encoding='utf-8') as hyp_file, \
                    open(references, 'r', encoding='utf-8') as ref_file:
                for hyp_line, ref_line in zip(hyp_file, ref_file):
                    self.params['hypothesis_reference']['hyp'].append(hyp_line.strip('\n').split(' '))
                    for line in ref_line.split('\t'):
                        ref_i = line.strip('\n').split(' ')
                        self.params['hypothesis_reference']['ref'].append([ref_i])
            if os.path.exists(self.params['FILE_PATH']):
                with open(self.params['FILE_PATH'], 'a') as file:
                    self.__write_corpus(file,
                                        self.params['model'],
                                        bleu_score.corpus_bleu(self.params['hypothesis_reference']['ref'],
                                                               self.params['hypothesis_reference']['hyp']),
                                        self.params['metric'])
            else:
                with open(self.params['FILE_PATH'], 'w') as file:
                    self.__write_corpus(file,
                                        self.params['model'],
                                        bleu_score.corpus_bleu(self.params['hypothesis_reference']['ref'],
                                                               self.params['hypothesis_reference']['hyp']),
                                        self.params['metric'])

    @staticmethod
    def __write_corpus(file, model: str, score: float, metric: str):
        """
        Helper Method which writes evaluations into the corresponding file

        :param score: The score achieved
        :param file: the opened file with correct attribute ('a' if it exists, otherwise 'w')
        :param model: The model which was used to generate the hypothesis
        :param score: The evaluated corpus score
        :return: This method write the result into the corresponding file
        """
        print('TimeStamp: {}\t'
              'Score: {:.12f}\t'
              'Metric: {}\t'
              'Model: {}'
              .format(datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"),
                      score,
                      metric,
                      model),
              file=file)

    @staticmethod
    def __write_single_or_batch_single(file, score: float, references: list, hypothesis: str):
        """
        Helper Method which writes evaluations into the corresponding file

        :param score: The score achieved
        :param file: the opened file with correct attribute ('a' if it exists, otherwise 'w')
        :param hypothesis: The hypothesis to be evaluated
        :param references: The references against which the hypothesis is being evaluated
        :return: This method write the result into the corresponding file
        """
        print(references)
        print('TimeStamp: {} \t'
              'Score: {:.12f} \t'
              'Hypothesis: {} \t'
              'Reference(s): {}'.
              format(datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"),
                     score,
                     hypothesis,
                     references),
              file=file)
