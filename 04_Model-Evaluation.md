## Model Evaluation

<img src="https://i.imgur.com/GI0pLVK.png" height="300" />

****************************************************
### Regression Models
1. Underfitting: - Model is not good enough to predict.
How to resolve: More Data, Train Longer

2. Overfitting: - Model memorizes and too much inclined with existing data. Predicting New unseen data will be a challange.
How to resolve: Early stop based on fit threshhold, More data, Sprinkle in some noise, regularization (data points smoothening), ensemble (combine models together), Drop some features.

### Residual Distribution | should be centerd to zero.
- if cetered around negetive value: prediction is too high
- if cetered around positive value: prediction is too low
<img src="https://docs.aws.amazon.com/machine-learning/latest/dg/images/mlconcepts_image4.png" height="300" />

### Error
<img src="https://i.imgur.com/iPUUP55.png" height="300" />

<img src="https://i.imgur.com/btT4DUz.png" height="300" />

### Binary Classificaiton Models

<img src="https://secureservercdn.net/198.71.233.197/l87.de8.myftpupload.com/wp-content/uploads/2016/09/table-blog.png" height="300" />

- **Accuracy:** The percent (ratio) of cases classified correctly.

- **Precision:** Accuracy of a predicted positive outcome in percent. | Fraud Detection
- **Recall:** Measure the strength of the model to predict a positive outcome. | Spam Checker
- **f1_score:** It is a combined metric. Harmonic mean of percision and recall.
**0(𝑏𝑎𝑑)≤f1_score≤1(good)**

- **Specificity:** Measure the strength of the model to predict a negetive outcome.

| Accuracy  | Precision | Recall | f1 score | Specificity |
|:---------:|:---------:|:------:|:--------:|:-----------:|
|TP + TN    |    TP     |   TP   |2(Pre*Rec)|   TN        |
|TP+TN+FP+FN|  TP+FP    | TP+FN  | (Pre+Rec)| TN+FP       |

#### AUC Graph by aws
<img src="https://i.imgur.com/c2XcKeg.png" height="300" />
<img src="https://i.imgur.com/HXz6Rgd.png" height="300" />

### Multiclass Classification
<img src="https://i.imgur.com/HXz6Rgd.png" height="300" />

*************************************************************************************
# AWS Advantage

## Provide clear tags of Test and Validation
<img src="https://i.imgur.com/c8IRiil.png" height="300" />

## Cloudwatch graph can help to plot accuracy 
<img src="https://i.imgur.com/0CPxT8n.png" height="300" />

## We can define what to send to clooudwatch
<img src="https://i.imgur.com/Q102ave.png" height="300" />
