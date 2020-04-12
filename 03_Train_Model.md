# Supervised

## Linear Learner | Predict quatitative value based on numeric input | Dense Continues Data
- Predict number (regression) or value below or above some threshhold (classification
- Adjust to Minimize error and get an equation for a line.
- To achive lowest possible on the map (error) uses Stochastic Gradient Descent
<img src="https://i.imgur.com/yYxfRz4.jpg" height="300" />
<img src="https://i.imgur.com/ZggmVjn.jpg" height="300" />

### In Regression
<img src="https://i.imgur.com/EFo6Mi6.png" height="300" />

### In Classification
To find local minimum and then global minimum
<img src="https://i.imgur.com/iWvbhX5.png" height="300" />

## Factorization Machine | Sparse data (with missing values)
- considers only pair-wise features
- csv is not supported
- Doesnt work for multiclass problem 
- only for prediction or binary classification
- needs lots of data
- does not perform well on dense data
- better work on CPU

Example: ClickStream ads data, movie recommendation

# Unsupervised
 
## Clustering K-Means | user defines identifying attribute
- Tabular data as input
- sagemaker use modified K-Means
- better on CPU

Example - analog to digital high and low as 0,1 , handwriting recognition black/white pixel as 0,1

**********************************************************************************************

## Classification K-Nearest Neighbor | user does not even defines identifying attributes
- predict value or classification to closest (avg value of nearest neighbour)
- KNN is lazy and momorizes
- kind of sterotyping

Example - Credit ratings, product recommendations for similar items

### AWS Image Algorithms
<img src="https://i.imgur.com/xpqNKQF.png" height="300" />

**********************************************************************************************
# Anomaly Detection

## Random Cut Forest
- provides anaomaly score

Example: Fraud Detection, IP Insights, Quality Control

**********************************************************************************************
# Text Analysis

## LDA - Latent Dirichlet Allocation
- Similar documents based on frequency of simialar words
Example: Recommendation article, Musical Influence Artist

## NTM - Neural Topic Model
- Perform topic modeling
Example: Recommendation article

## Seq2Seq
- Using vocab Language translation using Neural Network
Example - Speech to text AWS Polly

## BlazingText
- Natural Language Processing understanding context and semantic relationships
- Faster than traditional Word2Vec and FastText
Example - Amaozon Comprehend - Sentiment analysis, 
        - AWS Kendra - Enterprise search, 
        - Document Classification - Amazon Macie

## Object2Vec
- Identify relation between objects.
- sad, empathy, death as similar object
Example - Genere of the book, movie rating

**********************************************************************************************
## Reinforcement Learning
- Try to maximize reward
- Markoff decision process
Example - DeepRacer

**********************************************************************************************
## Forecasting
- Amazon DeepAR Lib for timeseries forecasting

**********************************************************************************************
## Ensemble Training | XGBoost
- Kinda swiss army knife for many regression, classification and ranking problems.
- Momory instance.

- Example: what price to ask for a house or car or phone?
    - CART Classificaitona and Regression Tress based on Locaiotn, Age, Size, rooms, walk score, climate etc.

Scenarios- Ranking, Fraud Detection
