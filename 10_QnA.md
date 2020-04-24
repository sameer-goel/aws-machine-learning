## ML Cheatsheets

<img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mathworks.com%2Fhelp%2Fstats%2Fmachine-learning-in-matlab.html&psig=AOvVaw3wR5LZh7UXAJjPipF6tajB&ust=1587698338171000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCKib067L_egCFQAAAAAdAAAAABAX" height="300" />

## Sagemaker data source options and speed
<img src="https://i.imgur.com/DbrpPWm.png" height="200" />

## Question on Dockerfile

We then create a Dockerfile with our dependencies and define the program that will be executed in SageMaker.

Remember when running a program to specify the entry point for Script Mode, set the SAGEMAKER_PROGRAM environmental variable. The script must be located in the /opt/ml/code folder.

```
FROM tensorflow/tensorflow:2.0.0a0

RUN pip install sagemaker-containers

# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM train.py

```

Ref: https://github.com/aws/sagemaker-containers/blob/master/README.rst

## [SageMaker Built-In Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)

### NLP
- BlazingText Algorithm - Semantically similar words, NLP Sentiment Analysis (Amazon Comprehend)
- Object2Vec Algorithm - find semantically similar objects such as questions
- Sequence-to-Sequence Algorithm - (Amazon Translate/Amazon Polly/Amazon Transcribe)

- Neural Topic Model (NTM) Algorithm - Topic Modeling
- Latent Dirichlet Allocation (LDA) Algorithm - Topic Modeling, product recommendations (Amazon Personalize)

### Computer Vision
- Image Classification Algorithm - multi-label classification (Amazon Rekognition)
- Object Detection Algorithm - object detection inside Box (Object Detection)
- Semantic Segmentation Algorithm -  (Object Detection)
- Instance segmentation - object detection as mask/figure - detecting and delineating each distinct object of interest appearing in an image. (Object Detection)

### Other
- DeepAR Forecasting Algorithm - RNN timeseries forecasting (Amazon Forecast)
- Factorization Machines Algorithm - click rate patterns (Amazon Kinesis Data Analytics)
- IP Insights Algorithm - detect anomalous IP address (Amazon GuardDuty)
- K-Means Algorithm - handwriting recognition black/white pixel as 0,1 (Amazon Textract)
- Principal Component Analysis (PCA) Algorithm - Dimention Reduction
- Random Cut Forest (RCF) Algorithm - (Amazon Kinesis Data Analytics)


### Regression
- Linear Learner Algorithm - either classification or regression problems
- XGBoost Algorithm ----- (Amazon Fraud Detector)
- K-Nearest Neighbors (k-NN) Algorithm - Credit ratings, product recommendations (Amazon Personalize)

Services:
- Amazon Polly - Text to Speech
- Amazon Transcribe - Speech to Text
- Amazon CodeGuru - code review
- Amazon Kendra - NLP powered search
- Amazon Lex - NLP bot

## Questions on Data Prep and Exploratory analysis

Q. 20% Numeric Data is missing
- Regression imputation
- Regularization imputation

Q. Classify good and bad sentences using LSTM
- Vectorize sentenses -> Tarform them to NUMERIC with PADDING -> Use sentences as input

Q. correlation between variable is -.9
- inversly propotional

__Q. Standardization__
Standardization (also called z-score normalization) transforms your data such that the resulting distribution has a mean of 0 and a standard deviation of 1.
<img src="https://i.imgur.com/GHwOoVO.gif" height="300" />

__Q. Scaling__
In scaling (also called min-max scaling), you transform the data such that the features are within a specific range e.g. [0, 1].
<img src="https://i.imgur.com/tpEqnnB.png" height="300" />

__Q. PCA and t-SNE Automatic feature extraction__
PCA is for linear models
t-SNE for non-linear models

Q. Stratified KFold?
- Ensures each fold is a good representative of the whole.

Q. Leave-one-out cross validation
- for small datasets

## Questions on Training

Q. Steps required for TRAINING JOB
1. __URL__ for source and destination S3 having data
2. __compute resource__ ML instance size specs
3. __Amazon Elastic Container Regitry Path__ for TRAINING CODE

Q. Dont like to wait while model is getting trained rather spend time on improving on model
- Use SageMaker Estimatoers in local mode

## Questions on Model Evaluation

Q. Overfitting avoidance techiniques
- more data, less features, regularization

Q. Validation has accuracy of 96% but Test data not?
- Need more data, shuffle it

## Questions on Linear learner Hyperparameter

Q. Models to tune hyperparameters
- GRID Search | exhaustive
- Random Search | random combination until desired outcome

Q. Sagemaker automated hyperparameter tuning
- uses methods like gradient descent, Bayesian optimization, and evolutionary algorithms to conduct a guided search for the best hyperparameter settings.

Q. Common hyperparameters
- Momentum
- Optimizers
- Activation Functions
- Dropout
- Learning Rate

Q. Image data set classification Model is taking too long to converge for more than 10 epochs
- Normalize images before training
- Add batch normalization

Q. __Loss__ when predictor_type for XGBoost
1. __regressor__, auto, squared_loss, absolute_loss, eps_insensitive_squared_loss, eps_insensitive_absolute_loss, quantile_loss, and huber_loss.
2. __binary_classifier__, auto,logistic, and hinge_loss.
3. __multiclass_classifier__, auto and softmax_loss

Q. Model Evaluation when __predictor_type__ is set to __binary_classifier__
- __accuracy__ — The model with the highest accuracy.
- __f_beta__ —The model with the highest F1 score. The default is F1.
- __precision_at_target_recall__ —The model with the highest precision at a given recall target.
- __recall_at_target_precision__ —The model with the highest recall at a given precision target.
- __loss_function__ —The model with the lowest value of the loss function used in training.

Q. Parameters and Default values
1. __normalize_label__ auto(only regression), true, or false | [-1,0,1]
2. __normalize_data__ auto, true, false | [-1,0,1]
3. __num_point_for_scaler__ (10,000) | number of data points to use for calculating normalization
4. __mini_batch_size__ (1000) | number of observations per mini-batch for the data iterator
5. __learning_rate__ (auto) | step size used by the optimizer for parameter updates.
6. __epochs__ (15)| The maximum number of passes over the training data.
7. __num_calibration_samples__ (auto) | number of observations from the validation dataset to use for model calibration
8. __early_stopping_tolerance__ 0.001 | relative tolerance to measure an improvement in loss
9. __target_precision__ (0.8) |  If binary_classifier_model_selection_criteria is recall_at_target_precision
10. __target_recall__ (0.8) |If binary_classifier_model_selection_criteria is recall_at_target_recall

## Questions of Classification Hyperparameter

Q.





Q. Loss function converges to different but stable values ,during multiple runs with identicle parameter 
- Reduce batch size, Decrease Learning rate/ Early stopping

Q. Data set is uneven for fraud Detection dataset
- SMOTE

Q. Common scaling techniques
- Mean/variance standardization [-1,0,1]
- MinMax [0,1]
- Maxabs
- Robust
- Normalizer

Q. Identify SageMaker supervised learning algorithms that are memory bound
- KNN, XGBoost

Q. Sagemkaer algorithms that can be parallelized
- 

Q. Algorithms that can be supervised and unsupervised
- Blazing Text

Q. algorithm that can be used both as a built-in-algorithm as well as a framework such as Tensorflow
- 

Q. To get inference for an entire dataset, you are developing a batch transform job using Amazon SageMaker High-level Python Library. Which method would you call so that the inferences are available for the entire dataset
- 

Q. Method calls you need to use to deploy the model
- 

Q. Visual Types
- Stock Price 

Q. activation function and their USEs
- Softmax
- Sigmoid
- RELU
- Tanh

Q. Kinesis Shard cound calculation

Q. Which PCA mode to use
- Regular - sparse data and a moderate number of observations and features. 
- Randomized - large number of observations and features.

Q. SageMaker algorithms support only the recordIO-protobuf file type for training data
- 

Q. Standardization vs Normalization vs Log Transformation
- 

Q. Trend vs Seasonality
- 

Q. After calling the create_training_job() method to start the training job, you would like to get a status about the progress of the training job
- 

Q. lack of reference historical data for similar phone model
-

Q. Distribution Types

Q. number of label columns in the content type

Q. Classification model using one of the Amazon SageMaker built-in algorithms and you want to use GPUs for both training and inference.
- 

Q. SageMaker support Regression vs Classification
- 



## Security

Q. S3 encryption options
- SSE-S3 - S3 manages key
- SSE-CMK - Cusomter manages key
- SSE-KMS - AWS manage data key, you manage master key

Q. SageMaker access security
- IAM roles having policies and conditional keys
- IAM Fedration
- Inside VPC, access only via PrivateLink

Q. SageMaker KMS Encryption options at Rest
- Notebooks
- Training Jobs
- SageMaker Endpoint
- S3 location for Trained Models

Q. SageMaker Encryption options in Transit
- Notebooks
- Training Jobs
- InterfaceEndpoint 
- Limited by Security group rule

- TLS 1.2 for Data in transit

Q. SageMaker Audit
- CloudTrail