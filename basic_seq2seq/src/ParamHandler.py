class ParamHandler:
    def __init__(self, variant, additional=None):
        self.params = {}
        self.additional_params = {}
        if variant not in ['char', 'word']:
            print("Not valid value for variant")
            exit(-1)
        self.params['BATCH_SIZE'] = 64
        self.params['EMBEDDING_DIM'] = 100
        self.params['EPOCHS'] = 15
        #self.params['GLOVE_FILE'] = "glove.6B.100d.txt"
        self.params['LATENT_DIM'] = 256
        self.params['MAX_DECODER_SEQ_LEN'] = 3633
        self.params['NUM_DECODER_TOKENS'] = 321
        self.params['MAX_ENCODER_SEQ_LEN'] = 3950
        self.params['NUM_ENCODER_TOKENS'] = 307
        self.params['MAX_NUM_SAMPLES'] = 2000000
        self.params['MAX_NUM_WORDS'] = 20000
        self.params['MAX_SENTENCES'] = 1000
        self.params['MAX_SEQ_LEN'] = 100
        self.params['P_DENSE_DROPOUT'] = 0.8
        self.params['VARIANT'] = variant
        self.additional_params = {}
        for i in range(len(additional)):
            self.additional_params["add" + str(i)] = additional[i]

    def param_summary(self):
        param_identifier = ""
        for param in sorted(self.params):
            param_identifier += '_' + str(self.params[param])
        if len(self.additional_params) > 0:
            param_identifier += "__"
            for param in sorted(self.additional_params):
                param_identifier += '_' + str(self.additional_params[param])
        return param_identifier
