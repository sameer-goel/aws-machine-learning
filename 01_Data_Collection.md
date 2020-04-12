
# Data Collections

## Data Formats supported by Sagemaker
1. File : Load from S3, CSV, JSON, Parquet, png, jpg
2. Pipe : Stream from S3, __recordIO-protobuf__

## Randomize
_Note Only non-timeseries data_

## Split
<img src="https://i.imgur.com/zUcDON7.png" height="300" />

## K Fold

Model one uses the first 25 percent of data for evaluation, and the remaining 75 percent for training. Model two uses the second subset of 25 percent (25 percent to 50 percent) for evaluation, and the remaining three subsets of the data for training, and so on. 

All k-fold validation rounds have roughly the same error rate. Otherwise, this may indicate that the data was not properly randomized before the training process.
<img src="https://docs.aws.amazon.com/machine-learning/latest/dg/images/image63.png" height="300" />
In a 4-fold cross-validation for a binary classification problem, each of the evaluations reports an area under curve (AUC) metric. You can get the overall performance measure by computing the average of the four AUC metrics.
**********************************************************************
