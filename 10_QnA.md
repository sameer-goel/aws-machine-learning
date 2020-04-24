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

- BlazingText Algorithm - Semantically similar words, NLP Sentiment Analysis (Amazon Comprehend)
- Object2Vec Algorithm - find semantically similar objects such as questions
- Neural Topic Model (NTM) Algorithm - Topic Modeling
- Latent Dirichlet Allocation (LDA) Algorithm - Topic Modeling, product recommendations (Amazon Personalize)
- Sequence-to-Sequence Algorithm - (Amazon Translate/Amazon Polly/Amazon Transcribe)

### Computer Vision
- Image Classification Algorithm - multi-label classification (Amazon Rekognition)
- Object Detection Algorithm - (Object Detection)

- DeepAR Forecasting Algorithm - RNN timeseries forecasting (Amazon Forecast)
- Factorization Machines Algorithm - click rate patterns (Amazon Kinesis Data Analytics)

- IP Insights Algorithm - detect anomalous IP address (Amazon GuardDuty)
- K-Means Algorithm - handwriting recognition black/white pixel as 0,1 (Amazon Textract)

- Principal Component Analysis (PCA) Algorithm - Dimention Reduction
- Random Cut Forest (RCF) Algorithm - (Amazon Kinesis Data Analytics)
- Semantic Segmentation Algorithm -  (Object Detection)

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

### Standardization
Standardization (also called z-score normalization) transforms your data such that the resulting distribution has a mean of 0 and a standard deviation of 1.
<img src="https://i.imgur.com/GHwOoVO.gif" height="300" />

### Scaling
In scaling (also called min-max scaling), you transform the data such that the features are within a specific range e.g. [0, 1].
<img src="https://i.imgur.com/tpEqnnB.png" height="200" />

# QnA

Q. Model is getting overtrained
- 

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

Q. Overfitting avoid
- more data, less features, regularization

Q. 