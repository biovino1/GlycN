**************************************************************************************************************
# N-linked Glycosylation Site
**************************************************************************************************************

This project uses several different datasets to train a 1D-CNN to predict if N-X-[S/T] sequons are glycosylated or not. Positive samples are taken from the N-GlycositeAtlas (http://nglycositeatlas.biomarkercenter.org/) and negative samples are taken from DeepLoc2.0 (https://services.healthtech.dtu.dk/services/DeepLoc-2.0/).

Protein sequences from these two datasets are combined for a total of 12,173 sequences, 6716 being glycosylated proteins and 6384 being non-glycosylated.
Entire sequences are embedded using the final hidden layer of ESM-2 (https://huggingface.co/facebook/esm2_t36_3B_UR50D) and the sequons from each one are used as inputs for the training of the model using pytorch framework, with glycosylated asparagine's being labeled as positive and non-glycosylated asparagine's in a sequon being labeled as negative.

5-fold cross validation and grid search are used to find the best hyperparameters for the model, which are then used to train the final model. The final model is then used to predict glycosylation sites in a testing set of asparagine embeddings held out from the training set.