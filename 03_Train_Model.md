# Supervised

## Linear Learner | Predict qualitative value based on numeric input | Dense Continues Data
- Predict number (regression) or value below or above some threshold (classification
- Adjust to Minimize error and get an equation for a line.
- To achieve lowest possible on the map (error) uses Stochastic Gradient Descent
<img src="https://i.imgur.com/yYxfRz4.jpg" height="300" />
<img src="https://i.imgur.com/ZggmVjn.jpg" height="300" />

### In Regression
<img src="https://i.imgur.com/EFo6Mi6.png" height="300" />

### Support Vector Machine
<img src="https://i.imgur.com/qjXfzl3.png" height="300" />

## Factorization Machine | Sparse data (with missing values)
- considers only pair-wise features
- csv is not supported
- Doesn't work for multi-class problem 
- only for prediction or binary classification
- needs lots of data
- does not perform well on dense data
- better work on CPU

Example: ClickStream ads data, movie recommendation
___________________
### In Classification
To find local minimum and then global minimum
<img src="https://i.imgur.com/iWvbhX5.png" height="300" />

## Decision Trees | also does feature selection in the process
<img src="https://i.imgur.com/z7IAV9q.png" height="300" />

### Random Forest | Collection of Decision Trees
<img src="https://i.imgur.com/X9IMY6B.png" height="300" />
____________________

### Classification K-Nearest Neighbor | user does not even defines identifying attributes
- predict value or classification to closest (avg value of nearest neighbor)
- KNN is lazy and motorizes
- kind of stereotyping

Example - Credit ratings, product recommendations for similar items

<img src="https://i.imgur.com/vYHuAjg.png" height="300" />

*******************************************************************************************

# Unsupervised Classification
 
## Clustering K-Means | user defines identifying attribute
- Tabular data as input
- sagemaker use modified K-Means
- better on CPU

Example - analog to digital high and low as 0,1 , handwriting recognition black/white pixel as 0,1

<img src="https://i.imgur.com/C3GKbNo.png" height="300" />

Elbow Plot, variation does not change from this point
<img src="https://i.imgur.com/VJh9R8J.png" height="300" />

**********************************************************************************************
# Anomaly Detection

## Random Cut Forest
- provides anaomaly score

Example: Fraud Detection, IP Insights, Quality Control

**********************************************************************************************
# AWS Image Algorithms

<img src="https://i.imgur.com/xpqNKQF.png" height="300" />

## LDA - Latent Dirichlet Allocation | UnSu | Classification  
__Example: Recommendation article, Musical Influence Artist, Topic Discovery, Automated Document tagging__
- Similar documents based on frequency of similar words
<img src="https://i.imgur.com/UEzK4b8.png" height="300" />

## NTM - Neural Topic Model
- Perform topic modeling
Example: Recommendation article

## Seq2Seq
- Using vocab Language translation using Neural Network
Example - Speech to text AWS Polly

## BlazingText
- Natural Language Processing understanding context and semantic relationships
- Faster than traditional Word2Vec and FastText
Example - Amazon Comprehend - Sentiment analysis, 
        - AWS Kendra - Enterprise search, 
        - Document Classification - Amazon Macie

## Object2Vec
- Identify relation between objects.
- sad, empathy, death as similar object
Example - Genre of the book, movie rating

**********************************************************************************************
## Reinforcement Learning
- Try to maximize reward
- Markoff decision process
Example - DeepRacer

**********************************************************************************************
## Forecasting
- Amazon DeepAR Lib for timeseries forecasting

**********************************************************************************************
## Ensemble Training | Bagging | XGBoost, AdaBoost
- Kinda swiss army knife for many regression, classification and ranking problems.
- Memory instance.

- Example: what price to ask for a house or car or phone?
    - CART Classification and Regression Tress based on Location, Age, Size, rooms, walk score, climate etc.

Scenarios- Ranking on an e-commerce website, Fraud Detection

**********************************************************************************************
## Tree Based Models

### 1. Decision trees
### 2. Ensemble Methods 
#### 2.1 Random Forest Bagging - Bootstrap Aggregating - Parallel Tree - Avoid underfitting
#### 2.2 Random Forest Bagging - Boosting - Sequential - Avoid overfitting
Learn from previous

**********************************************************************************************

## Transfer Learning
Teach maths and then teach data science
<img src="https://i.imgur.com/KgtD3dV.png" width="200" />