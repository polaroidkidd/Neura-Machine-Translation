from metrics.Bleu import Bleu

hypothesis_file = "C:/Users/Nicolas/Desktop/deu_val_data_not_seen.pred"
references_file = "C:/Users/Nicolas/Desktop/deu_val_data_not_seen.de"

bleu_evaluator = Bleu("model_2_token_also_at_encoder_unk", 'bleu', timestamp=True)
bleu_evaluator.evaluate_hypothesis_batch_single(hypothesis_file, references_file)
