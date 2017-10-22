from nltk.translate import bleu_score
from datetime import datetime
from metrics.BaseMetric import BaseMetric
import os


class Bleu(BaseMetric):
    def __init__(self, model, timestamp=False):
        """

        :param model: The name of the model.
        :param timestamp: if set to true the file name will be appended by a time stamp
        """
        BaseMetric.__init__(self)
        self.params['model'] = model

        if timestamp:
            self.params['timestamp'] = datetime.strftime(datetime.now(), "%Y-%m-%d__%H-%M-%S")
        else:
            self.params['timestamp'] = ''
        self.params['hypothesis_reference'] = {
            'hyp': None,
            'ref': None
            }

        self.params['RESULT_DIR'] = '../../../evaluations/' + self.params['model']
        if not os.path.exists(self.params['RESULT_DIR']):
            os.mkdir(self.params['RESULT_DIR'])
        self.params['FILE_NAME'] = model + '_' + self.params['timestamp'] + '_BLEU.txt'
        self.params['FILE_PATH'] = self.params['RESULT_DIR'] + '/' + self.params['FILE_NAME']

    def evaluate_hypothesis_single(self, hypothesis: str, references: list):
        """
        Evaluates predictions via the bleu score metric
        :param references: A list of references against which the predicted string is measured. If only one reference
        is available this method accepts it as a string and converts it to a single-item list
        :param hypothesis: The predicted string(s)
        :type hypothesis: list
        :type references: list
        """

        if not type(references) == list:
            references = [references]
        self.params['hypothesis_reference']['hyp'] = hypothesis
        self.params['hypothesis_reference']['ref'] = references
        if os.path.exists(self.params['FILE_PATH']):
            with open(self.params['FILE_PATH'], 'a') as file:
                self.write_to_file(file, hypothesis, references)
        else:
            with open(self.params['FILE_PATH'], 'w') as file:
                self.write_to_file(file, hypothesis, references)

    def evaluate_hypothesis_batch_single(self, references, hypothesis):
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
            with open(hypothesis, 'r') as hyp_file, open(references, 'r') as ref_file:
                for references_line, hypothesis_line in zip(ref_file, hyp_file):
                    self.evaluate_hypothesis_single(hypothesis_line.strip('\n'), references_line.strip('\n'))

    def evaluate_hypothesis_corpus(self, hypothesis, references):
        pass

    def write_to_file(self, file, hypothesis, references):
        print('TimeStamp: {} \t'
              'Score: {:.12f} \t'
              'Hypothesis: {} \t'
              'Reference(s): {}'.
              format(datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"),
                     bleu_score.sentence_bleu(hypothesis=self.params['hypothesis_reference']['hyp'],
                                              references=self.params['hypothesis_reference']['ref']),
                     hypothesis, references), file=file)
