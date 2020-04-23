## ML Cheatsheets

<img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mathworks.com%2Fhelp%2Fstats%2Fmachine-learning-in-matlab.html&psig=AOvVaw3wR5LZh7UXAJjPipF6tajB&ust=1587698338171000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCKib067L_egCFQAAAAAdAAAAABAX" height="300" />

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

- BlazingText Algorithm - NLP Sentiment Analysis (Amazon Comprehend)
- DeepAR Forecasting Algorithm - RNN timeseries forecasting (Amazon Forecast)
- Factorization Machines Algorithm - click rate patterns (Amazon Kinesis Data Analytics)
- Image Classification Algorithm - multi-label classification (Amazon Rekognition)
- IP Insights Algorithm - detect anomalous IP address (Amazon GuardDuty)
- K-Means Algorithm - handwriting recognition black/white pixel as 0,1 (Amazon Textract)
- K-Nearest Neighbors (k-NN) Algorithm - Credit ratings, product recommendations (Amazon Personalize)
- Latent Dirichlet Allocation (LDA) Algorithm - Topic Modeling, product recommendations (Amazon Personalize)
- Linear Learner Algorithm - either classification or regression problems
- Neural Topic Model (NTM) Algorithm - Topic Modeling
- Object2Vec Algorithm - 
- Object Detection Algorithm - (Object Detection)
- Principal Component Analysis (PCA) Algorithm - Dimention Reduction
- Random Cut Forest (RCF) Algorithm - (Amazon Kinesis Data Analytics)
- Semantic Segmentation Algorithm -  (Object Detection)
- Sequence-to-Sequence Algorithm - (Amazon Translate/Amazon Polly/Amazon Transcribe)
- XGBoost Algorithm ----- (Amazon Fraud Detector)

Services:
- Amazon Polly - Text to Speech
- Amazon Transcribe - Speech to Text
- Amazon CodeGuru - code review
- Amazon Kendra - NLP powered search
- Amazon Lex - NLP bot
