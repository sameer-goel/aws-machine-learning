## Model Evaluation

<img src="https://i.imgur.com/GI0pLVK.png" height="300" />

****************************************************
### Regression Models
1. Underfitting: - Model is not good enough to predict.
2. Overfitting: - Model memorizes and too much inclined with existing data. Predicting New unseen data will be a challange.

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