from models.keras_tut_original import KerasTutSeq2Seq

model = KerasTutSeq2Seq()
model.setup_inference()
model.predict("Go away!")