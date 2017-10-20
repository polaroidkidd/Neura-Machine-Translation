import sys

from helpers import model_repo

if len(sys.argv) > 1:
    model = model_repo.argument_model_selection(sys.argv[1])
else:
    model = model_repo.interactive_model_selection()

model.start_training()
