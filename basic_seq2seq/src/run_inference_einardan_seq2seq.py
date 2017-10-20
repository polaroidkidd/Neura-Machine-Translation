from inference.einardanSeq2Seq import Einardan_Inference
from helpers.chkpt_to_three_models import Chkpt_to_Models

chkpt_to_models = Chkpt_to_Models()
chkpt_to_models.start()

inference = Einardan_Inference()
inference.start()
