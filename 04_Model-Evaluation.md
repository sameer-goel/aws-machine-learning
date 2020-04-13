## Model Evaluation

<img src="https://i.imgur.com/GI0pLVK.png" height="300" />

****************************************************
### Regression Models
1. Underfitting: - Model is not good enough to predict.
How to resolve: More Data, Train Longer

2. Overfitting: - Model memorizes and too much inclined with existing data. Predicting New unseen data will be a challange.
How to resolve: Early stop based on fit threshhold, More data, Sprinkle in some noise, regularization (data points smoothening), ensemble (combine models together), Drop some features.

<img src="https://i.imgur.com/v1iorkD.png" height="300" />

<img src="https://i.imgur.com/iPUUP55.png" height="300" />

### Classificaiton Models

<img src="https://secureservercdn.net/198.71.233.197/l87.de8.myftpupload.com/wp-content/uploads/2016/09/table-blog.png" height="300" />

- **Accuracy:** The percent (ratio) of cases classified correctly.

- **Precision:** Accuracy of a predicted positive outcome in percent.
- **Recall:** Measure the strength of the model to predict a positive outcome.
- **f1_score:** It is a combined metric. Harmonic mean of percision and recall.
**0(𝑏𝑎𝑑)≤f1_score≤1(good)**

- **Specificity:** Measure the strength of the model to predict a negetive outcome.

| Accuracy  | Precision | Recall | f1 score | Specificity |
|:---------:|:---------:|:------:|:--------:|:-----------:|
|TP + TN    |    TP     |   TP   |2(Pre*Rec)|   TN        |
|TP+TN+FP+FN|  TP+FP    | TP+FN  | (Pre+Rec)| TN+FP       |

*************************************************************************************
# AWS Advantage

## Provide clear tags of Test and Validation
<img src="https://i.imgur.com/c8IRiil.png" height="300" />

## Cloudwatch graph can help to plot accuracy 
<img src="https://i.imgur.com/0CPxT8n.png" height="300" />

## We can define what to send to clooudwatch
<img src="https://i.imgur.com/Q102ave.png" height="300" />
