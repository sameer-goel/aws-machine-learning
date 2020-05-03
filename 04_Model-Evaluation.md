## Model Evaluation

## Offline Validation
Both k-fold and backtesting with historic data are offline validation methods

## Online Validation
with real world data

****************************************************
## Regression Models, look for generalization


<img src="https://i.imgur.com/GI0pLVK.png" height="300" />

1. Under-fitting: - Model is not good enough to predict.
How to resolve: More Data, __Train Longer__, Add more features to dataset

2. Over-fitting: - Model memorizes and too much inclined with existing data. Predicting New unseen data will be a challenge.
How to resolve: __Early stop__ based on fit threshold, More data, Sprinkle in some noise, regularization (data points smoothening), ensemble (combine models together), Drop some features.
<img src="https://i.imgur.com/e7rAimx.png" height="400" />

## Bias and Variance | sum of squares
<img src="https://i.imgur.com/6SjMa9m.png" height="200" />

### Introducing Regularization to adjust variance at cost of bias to avoid over-fitting
- L2: Ridge regression - increase bias - introduce error to avoid overfitting
- L1: Lasso regression - increase alpha to reduce the slope = ZERO
<img src="https://i.imgur.com/tnjCyRI.png" height="400" />

### Residual Distribution | should be centered to zero.
- if centered around negative value: prediction is too high
- if centered around positive value: prediction is too low
<img src="https://docs.aws.amazon.com/machine-learning/latest/dg/images/mlconcepts_image4.png" height="300" />

### Regression Errors Type
<img src="https://i.imgur.com/iPUUP55.png" height="300" />

<img src="https://i.imgur.com/btT4DUz.png" height="300" />

### Gradient Descent
|![](https://i.imgur.com/Vaw5vuC.png)|![]()|

Step should not be too large (miss minimum) or too small (take longer)
<img src="https://i.imgur.com/gBpuE6A.png" height="300" />

## Binary Classification Models

<img src="https://secureservercdn.net/198.71.233.197/l87.de8.myftpupload.com/wp-content/uploads/2016/09/table-blog.png" height="300" />

- **Accuracy:** The percent (ratio) of cases classified correctly.

- **Precision:** Accuracy of a predicted positive outcome in percent.| reduce false +ve | Spam Checker, Anomaly detection

- **Recall/Sensitivity/TPR:** Measure the strength of the model to predict a positive outcome. | reduce false -ve (FN)
Fraud Detection, Medical Diagnostics, COVID19 Test

- **Specificity/TNR:** Measure the strength of the model to predict a negative outcome.
reduce FP, Spam detector, Explicit Content Blog, Catch just Fish

- **f1_score:** It is a combined metric. Harmonic mean of precision and recall.
**0(𝑏𝑎𝑑)≤f1_score≤1(good)**

| Accuracy  | Precision | Recall | f1 score | Specificity |
|:---------:|:---------:|:------:|:--------:|:-----------:|
|TP + TN    |    TP     |   TP   |2(Pre*Rec)|   TN        |
|TP+TN+FP+FN|  TP+FP    | TP+FN  | (Pre+Rec)| TN+FP       |

<img src="https://miro.medium.com/max/908/1*t_t7cMq3FGqDk6gbwfA4EA.png" height="300" />

## ROC/AUC
<img src="https://i.imgur.com/UA95C7d.png" height="300" />

#### AUC Graph by aws
<img src="https://i.imgur.com/c2XcKeg.png" height="300" />

## Gini Impurity
Look for Lowest Impurity
<img src="https://i.imgur.com/MQPGZA6.png" height="300" />

## MultiClass Classification
<img src="https://i.imgur.com/HXz6Rgd.png" height="300" />

*************************************************************************************
# AWS Advantage

## Provide clear tags of Test and Validation
<img src="https://i.imgur.com/c8IRiil.png" height="300" />

## CloudWatch graph can help to plot accuracy 
<img src="https://i.imgur.com/0CPxT8n.png" height="300" />

## We can define what to send to clooudwatch
<img src="https://i.imgur.com/Q102ave.png" height="300" />

****************************************************************************************
### Linear Regression
Best fitting line
`y=w_0+w_1 x`

### Log-loss (Binary Cross-Entropy)
Check performance of binary classifier
`LogLoss=−(y∗log(p)+(1−y)∗log⁡(1−p))`

*************************************************************************************

__Tips:__

Q. If loss function is settling on a similar value, how to improve model
A. A learning rate can be too large that it cannot find the the true global minimum. Decreasing the learning rate allows the training process to find lower loss function floors but it can also increase the time needed for convergence.

Q. Linear Learner model should be tuned on?
A. AWS recommends tuning the model against a validation metric instead of a training metric. 

Q. Data is highly imbalanced like credit card fraud detection
- FN is big deal and should be less, Falsely identifying fraud is negative.
- RECALL (Type 1 error) is important here
<img src="https://i.imgur.com/rFRl5jR.png" width="400" />

Q. Model is ok to predict to get false cases but
- FP is big deal and should be less, Falsely identifying spam as positive.
<img src="https://i.imgur.com/UqWl3Pg.png" width="400" />