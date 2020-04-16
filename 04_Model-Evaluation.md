## Model Evaluation

<img src="https://i.imgur.com/GI0pLVK.png" height="300" />

## Offline Validation
Both k-fold and backtesting with historic data are offline validation methods

## Online Validation
with real world data

****************************************************
## Regression Models, look for generalization

1. Underfitting: - Model is not good enough to predict.
How to resolve: More Data, __Train Longer__, Add more features to dataset

2. Overfitting: - Model memorizes and too much inclined with existing data. Predicting New unseen data will be a challange.
How to resolve: __Early stop__ based on fit threshhold, More data, Sprinkle in some noise, regularization (data points smoothening), ensemble (combine models together), Drop some features.

### Residual Distribution | should be centerd to zero.
- if cetered around negetive value: prediction is too high
- if cetered around positive value: prediction is too low
<img src="https://docs.aws.amazon.com/machine-learning/latest/dg/images/mlconcepts_image4.png" height="300" />

### Error
<img src="https://i.imgur.com/iPUUP55.png" height="300" />

<img src="https://i.imgur.com/btT4DUz.png" height="300" />

### Gradient Descent
|![](https://i.imgur.com/Vaw5vuC.png)|![]()|

Step should not be too large (miss ninimum) or too small (take longer)
<img src="https://i.imgur.com/gBpuE6A.png" height="300" />

## Binary Classificaiton Models

<img src="https://secureservercdn.net/198.71.233.197/l87.de8.myftpupload.com/wp-content/uploads/2016/09/table-blog.png" height="300" />

- **Accuracy:** The percent (ratio) of cases classified correctly.

- **Precision:** Accuracy of a predicted positive outcome in percent.| reduce false +ve | Spam Checker, Anomaly detection
- **Recall:** Measure the strength of the model to predict a positive outcome. | reduce false -ve | Fraud Detection
- **f1_score:** It is a combined metric. Harmonic mean of percision and recall.
**0(𝑏𝑎𝑑)≤f1_score≤1(good)**

- **Specificity:** Measure the strength of the model to predict a negetive outcome.

| Accuracy  | Precision | Recall | f1 score | Specificity |
|:---------:|:---------:|:------:|:--------:|:-----------:|
|TP + TN    |    TP     |   TP   |2(Pre*Rec)|   TN        |
|TP+TN+FP+FN|  TP+FP    | TP+FN  | (Pre+Rec)| TN+FP       |

<img src="https://miro.medium.com/max/908/1*t_t7cMq3FGqDk6gbwfA4EA.png" height="300" />

#### AUC Graph by aws
<img src="https://i.imgur.com/c2XcKeg.png" height="300" />

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

*************************************************************************************

__Tips:__

Q. If loss funciton is settling on a similar value, how to imporve model
A. A learning rate can be too large that it cannot find the the true global minimum. Decreasing the learning rate allows the training process to find lower loss function floors but it can also increase the time needed for convergence.

Q. Linear Learner model should be tuned on?
A. AWS recommends tuning the model against a validation metric instead of a training metric. 


***********
### Linear Regression
Best fitting line
`y=w_0+w_1 x`

### Log-loss (Binary Cross-Entropy)
Check peformance of binary classifier
`LogLoss=−(y∗log(p)+(1−y)∗log⁡(1−p))`