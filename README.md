**************************************************************************************************************
# GlycN
**************************************************************************************************************

This project embeds glycosylated and non-glycosylated proteins using ESM-2 (https://huggingface.co/facebook/esm2_t36_3B_UR50D) and trains a 1D-CNN to predict if N-X-[S/T] sequons are glycosylated or not.

**************************************************************************************************************
# DATA
**************************************************************************************************************

This project uses several different datasets to train the model. Positive samples are taken from the N-GlycositeAtlas (http://nglycositeatlas.biomarkercenter.org/) where glycosylated residues are index and negative samples are taken from DeepLoc2.0 (https://services.healthtech.dtu.dk/services/DeepLoc-2.0/) which indicate in which
organelles sequences came from. Proteins from mitrochondria and nucleus are unable to be glycosylated and are used as negative samples.

Protein sequences from these two datasets are combined for a total of 12,173 sequences, 6716 being glycosylated proteins and 6384 being non-glycosylated. Entire sequences are embedded using ESM-2 and the asparagine residues are extracted and used as inputs for the training of the model using pytorch framework, with glycosylated asparagine's being labeled as positive and non-glycosylated asparagine's in a sequon being labeled as negative.

5-fold cross validation and grid search are used to find the best hyperparameters for the model, which are then used to train the final model. The final model is then used to predict glycosylation sites in a testing set of asparagine embeddings held out from the training set.

**************************************************************************************************************
# RESULTS
**************************************************************************************************************

The architecture of the model can be found in model.py and the optimized hyperparameters can be found in config.yaml. 80% of the data is used for cross-validation and training and 20% is used for testing (MMseqs2 can be used to cluster sequences, although this does not change the average results by much).

 With 5-fold cross-validation, the model acheived an average accuracy of 0.917, precision of 0.926, recall of 0.906, F1 of 0.916, AUC of 0.917 and MCC of 0.835. 
 
 Trained on the entire training set, the model acheived a testing accuracy of of 0.923, precision of 0.946, recall of 0.900, F1 of 0.922, AUC of 0.923, and MCC of 0.846

**************************************************************************************************************
# FUTURE DIRECTIONS
**************************************************************************************************************

While this model nears the accuracy of current state-of-the-art N-glycosylation prediction, there is much more room to improve in the area of O-glycosylation prediction. This is much more difficult to predict and will require different datasets which are currently being explored.