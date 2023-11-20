**************************************************************************************************************
# N-linked Glycosylation Site and Pattern Prediction
**************************************************************************************************************

This project uses several different datasets to train a 1D-CNN to predict if N-X-[S/T] sequons are glycosylated or not. Positive samples are taken from the N-GlycositeAtlas (http://nglycositeatlas.biomarkercenter.org/) and negative samples are taken from DeepLoc2.0 (https://services.healthtech.dtu.dk/services/DeepLoc-2.0/).

Sequences are embedded using the final hidden layer of ESM-2 (https://huggingface.co/facebook/esm2_t36_3B_UR50D) and used as inputs for the training of the model using pytorch framework.