## ML Cheat sheets

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
## Question on SageMaker Python SDK

Q. Using own tensor flow code.
- Use tensorflow in SageMaker and edit cod ein SageMaker Python SDK


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
- Principal Component Analysis (PCA) Algorithm - Dimension Reduction
- Random Cut Forest (RCF) Algorithm - (Amazon Kinesis Data Analytics)

### Regression
- Linear Learner Algorithm - either classification or regression problems
- XGBoost Algorithm ----- (Amazon Fraud Detector)
- K-Nearest Neighbors (k-NN) Algorithm - Credit ratings, product recommendations (Amazon Personalize)

### Services:
- Amazon Polly - Text to Speech
- Amazon Transcribe - Speech to Text
- Amazon CodeGuru - code review
- Amazon Kendra - NLP powered search
- Amazon Lex - NLP bot

Q. Identify SageMaker supervised learning algorithms that are memory bound
- Most Amazon SageMaker algorithms have been engineered to take advantage of GPU computing for training. Despite higher per-instance costs, GPUs train more quickly, making them more cost effective. 
- Exceptions, such as __XGBoost__, __Random Cut Forest__, __LDA__

`[Ref] Common parameters for built-in algorithms  - https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html`

Q. Single instance only Algorithms
- CPU/GPU Single instance only: Object2Vec, Blazing Text, KNN, K-means
- CPU Single instance only: LDA
- GPU Single instance only: Seq2Seq, Semantic Segmentation

Q. Algorithms that just accept recordIO-recordIO-protobuf
- Factorization Machine
- Seq2Seq

Q. Algorithms that don't accept recordIO-recordIO-protobuf
- DeepAR - JSON or Parquet
- Object2Vec - JSON
- IP Insights - CSV
- XGBoost - CSV

- Semantic Segmentation - Image files
- Object Detection - recordIO MXNet
- Blazing Text - Text file, 1 sentence per line with space separated token

Q. All other are accept recordIO-recordIO-protobuf and CSV: 
- Image classification, , k-means, K-NN, PCA, Linear learner, Neural Topic Modeling, LDA, RCF

Q. Sagemaker algorithms that can be parallelized
- 

Q. Algorithms that can be supervised and unsupervised
- Blazing Text
    - Unsupervised Word2Vec 
    - Supervised TextClassification 

Q. algorithm that can be used both as a built-in-algorithm as well as a framework such as Tensorflow
- __XGBoost__

Q. While using the K-means SageMaker algorithm, which strategies are available to determine how the initial cluster centers are selected.
- Random
- k-means++

Q. Amazon SageMaker provides Neo container images for?
- XGBoost and Image Classification models

## Questions on Exploratory analysis

Q. Visual Types
- Stock Price : Scatter Plot, Box Plot

## Questions on Data Prep

Q. Month field is Ordinal or Nominal
- Month is Nominal so One-Hot Encoder

Q. 20% Numeric Data is missing
- Regression imputation
- Regularization imputation

Q. 20% Categorical data is missing
- use KNN

Q. Classify good and bad sentences using LSTM
- Vectorize sentences -> Transform them to NUMERIC with PADDING -> Use sentences as input

Q. correlation between variable is -.9
- inversely proportional

Q. Data set is uneven for fraud Detection dataset
- SMOTE

__Q. Standardization__
Standardization (also called z-score normalization) transforms your data such that the resulting distribution has a mean of 0 and a standard deviation of 1.
<img src="https://i.imgur.com/GHwOoVO.gif" height="300" />

__Q. Scaling__
- In scaling (also called min-max scaling), you transform the data such that the features are within a specific range e.g. [0, 1].
<img src="https://i.imgur.com/tpEqnnB.png" height="300" />

Q. [PCA and t-SNE](https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b) Automatic feature extraction
- PCA is for linear models
- t-SNE for non-linear models

Q. Stratified KFold?
- Ensures each fold is a good representative of the whole.

Q. Leave-one-out cross validation
- for small datasets

Q. Which aws service can transform data output to RecordIO-Protobuf
- Apache Spark EMR Cluster. (Glue or Kinesis doesn't work)

## Questions on Training

Q. Steps required for TRAINING JOB
1. __URL__ for source and destination S3 having data
2. __compute resource__ ML instance size specs
3. __Amazon Elastic Container Registry Path__ for TRAINING CODE

Q. Don't like to wait while model is getting trained rather spend time on improving on model
- Use SageMaker Estimators in local mode

## Questions on Model Evaluation

Q. Overfitting avoidance techniques
- more data, 
- less features
- early stopping
- dropout regularization

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



## Questions on auto hyperparameter tuning

Q. Sagemaker is consuming more resources and costing high.
- Use less concurrency
- use LOGarithmic scales on parameter ranges
- running 1 training job at a time achieves the best results with the least amount of compute time.

Q. Loss function converges to different but stable values ,during multiple runs with identical parameter 
- Reduce batch size, Decrease Learning rate/ Early stopping

Q. Common scaling techniques
- Mean/variance standardization [-1,0,1]
- MinMax [0,1]
- Maxabs
- Robust
- Normalizer


Q. To get inference for an entire dataset, you are developing a batch transform job using Amazon SageMaker High-level Python Library. Which method would you call so that the inferences are available for the entire dataset
- 

Q. Method calls, you need to use to deploy the model
- 

Q. activation function and their USEs
- Softmax
- Sigmoid
- RELU
- Tanh

Q. Kinesis Shard count calculation

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

## Questions on text matrix

Q. tf-idf matrix for  { Hold please }, { Please try again }, { Please call us back }.
- 3x3 because 3 sentence x 3 trigrams

## Security

Q. S3 encryption options
- SSE-S3 - S3 manages key
- SSE-CMK - Customer manages key
- SSE-KMS - AWS manage data key, you manage master key

Q. SageMaker access security
- IAM roles having policies and conditional keys
- IAM Federation
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

Q. Encrypt data using GLUE
- Encrypt __Data Catalog__
- Encrypt __Job bookmark__
- SSE-S3 or SSE-KMS settings that is passed to __ETL Job__
- Also do, S3 Encryption, CloudWatch log encryption 

Q. Secure S3 <-> RDS <-> Glue communication
- S3  <-> RDS : VPC Gateway Endpoint to access to S3 from RDS
- RDS <-> Glue: AWS Glue sets up elastic network interfaces that enable your jobs to connect securely to RDS within your VPC

Q. Sagemaker to Read encrypted data in S3
- Notebook instance role to be associated with KMS Key.
- Ensure s3 bucket has SSE-KMS encryption

## Questions of Sagemaker

Q. You wish to use Apache spark to pre-process the data for XGBoost model.
- use sagemaker_pyspark and XGBoost sagemaker estimator

Q. Data to be queries is stores in S3.
- Easy way: Use Glue and Athena
- Cost effective way: Use presto on spot instance

Q. Transfer data from local machine into your AWS data repository for Semantic Segmentation.
- Host the dataset in Amazon S3 and storing it in 2 channels. 
- 1st channel for train and 2nd for validation, in 4 directories, 
- 2 for images and 2 for annotations. 
- Use a label map that describes how the annotation mappings are established.

Q. Use custom lib for transformation with Glue Pipeline.
- Upload lib as .zip in S3, include s3 link as script lib and job parameter.

Q. Test ML model in Prod.
- 2 models on SINGLE endpoint
- Route % of traffic to each for evaluation of best one
- Route 100% to the model having better performance.

Q. IAM Policy required to create Models
- iam:PassRole action is needed for the Amazon SageMaker action sagemaker:CreateModel

## Questions on Model Deploying

Q. Validate a model Offline
- Use historical data - __backtesting__ with __HOLDOUT Set__ typically 10-20% of Training Data.
- K Fold validation

Q. Validation online (Production variant) | Testing with small % of live data
- Multiple Models are deployed on single endpoint and then small portion of the live traffic goes to those models that you want to validate to find out best performing one.

Q. Deploy Methods?
- Hosting
- Batch Transform

Q. Hosting
- SageMaker SDK 
    - .deploy()
- AWS SDK
    1. __CreateModel__
    2. __CreateEndpointConfig__
    3. __CreateEndpoint__

Q. How to make change to inference pipeline.
- deploy a new one using __UpdateEndpoint__ API
- pipeline is immutable

Q. Batch Transform
- SageMaker SDK
    - transformer.transform()
- AWS SDK
    - create_transform_job()

Q. Amazon Elastic Inference? 
- allows you to attach low-cost GPU-powered acceleration to EC2 and Sagemaker instances or Amazon ECS tasks, 
- to reduce the cost of running deep learning inference by up to 75%. 
- Supports __TensorFlow__, __Apache MXNet__, __PyTorch__ and __ONNX__ models

Q. Scaling SageMaker Endpoint
- TargetTrackingScalingPolicyConfiguration
- reduce Cooldown period for aggressive scaling 

Q. Want to add low cost GPU
- RE-Deploy endpoint with Elastic Inference

Q. IoT <-> Streaming data <-> Inference endpoint
1. IoT Core get data from IoT device as MQTT
2. Kinesis stream to fetch data and send to lambda for serialization
3. Lambda Serializes and send to Inference endpoint
4. Inference endpoint De-Serializes back

Q. ML Lib for capturing IoT data
- MLeap for serialization
- MLib for building Model
- SparkML serving container to Deploy this pipeline

Q. What is Horovod? 
- It is a framework allowing a user to distribute a deep learning workload among multiple compute nodes and take advantage of inherent parallelism of deep learning training process.


# Exploratory data analysis

Q. fraud detection data is un-even
- SMOTE but it creates almost identical/duplicate records
- GANs creates more unique but closely matching

Q. Streaming data for RCF requires pre and post steps via ML model
- Use INFERENCE Pipeline.
- Use Batch Transform

# Modeling Domain

Q. Algo for Recommendation System 
-  KNN

Q. Segment customer based on their spending habits 
- K-Means

Q. Email phishing 
- N-Gram

Q. Starting with std pre-trained market model and then further training it 
- Transfer Learning

Q. Hyperparameter that governs how quickly model adepts new or changing data.
- Set Learning rate at high value (0.0 - 1.0)

Q. Bayesian vs Random Hyperparameter Tuning
- __Bayesian__ search treats hyperparameter tuning like a [regression] problem. Given a set of input features (the hyperparameter), hyperparameter tuning optimizes a model for the metric that you choose. 
- __Random__ search, hyperparameter tuning chooses a random combination of values from within the ranges that you specify for hyperparameter for each training job it launches

Q. Steps to pass tuning job settings in hyperparameter tuning as JSON.
- specify __value__ of HyperParameterTuningJobConfig
- specify __range__
- specify __objective__

Q. Ways to monitor metrics of training the model.
- AWS management Console
- Python SDK APIs
- CloudWatch console for visualizing time-series

Q. How to specify metrics for logging
- specify REGEX pattern for metrics
- CloudWatch to visualize these automatically parsed metrics

Q. Visualize K-means metrics tuning job
- specify valid metric eg. test:msd
- use module __sagemaker.analytics__
      import __TrainingJobAnalytics__

Q. Find most predictive booster feature of XGBoost
- __booster = gbtree__ using get_score with __importance_type = total_gain__

Q. Glue ML to FindMatches to get rid of duplicates.
- precision-recall to __'precision'__ (because we are minimizing false positive, means should not flag any distinct record as duplicate)
- accuracy-cost parameter to __'accuracy'__ (setting to __lower_cost__ might be a compromise with accuracy)

Q. Glue ML to FindMatches input file requirement.
- Labeling file in UTF-8 without BOM(byte order mark)

Q. Glue DataFrames, DynamicFrames, Dynamic Records
- DataFrame     : Popular, requires schema for ETL
- DynamicFrames : Each record is self describing so sno schema required
- DynamicRecord : Logical record within DynamicFrames

Q. Handle variety schema structure incoming records in GLUE.
- Transform using DynamicFrames to pass data from transform to transform

# Data Engineering Domain

Q. Automate wrong labeled data by Ground Truth
- Annotation Consolidation allows to combine annotations of multiple workers to produce probabilistic model.

Q. Ground Truth Automated Data Labeling

- Image classification (Single label)
- Semantic segmentation
- Bounding box
- Text classification

Q. Deployed model in production requires to get inferences on ENTIRE Dataset and 
don't need PERSISTENT endpoint.
- SageMaker BATCH Transform

Q. reason why Firehose data is rejected records sent by lambda
- Firehose requires 1. record id 2. data params 3. result params, which might be missing.

### TIPs and Misc

- CNN - Convolutional Neural Networks are mostly for __image and signal__ processing. 
- RNN - Recurrent Neural Networks are mostly for __text or speech__ use-cases where sequence prediction is key.
- Binary Classification - AUC->1 is metric to look for. | logistic or hinge_loss is objective.
- MultiClass Classification - Macro F1 score to look for. | muti:softmax is objective.
- Seq2Seq can is used for translation and summarization

`Important`
- XGBoost requires only numerical features plus this Tree-based algo can handle features with different scale (no need to normalize/standardize)
- Tree based algorithm can have lower and upper bound limit it can predict for regression. (random forest, xgboost)

- When tuning the model, don't use Test data, use validation data.
- Seq2Se2 model needs RecordIO-protobuf with INTEGER tokens