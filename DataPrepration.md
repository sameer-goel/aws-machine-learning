## Data Prepration

Most important and time consuming process of Data Science like Tex cleaning, Missing Values, Outliers, OneHot encoding, Unform data types.

<img src="https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fgilpress%2Ffiles%2F2016%2F03%2FTime-1200x511.jpg" width="600" />

## Start with Descriptive Statistics

* Overall statistics
    * Number of instances (i.e. number of rows)
    * Number of attributes (i.e. number of columns)
* Attribute statistics (univariate or single variable)
    * Statistics for **numeric** attributes (mean, variance, etc.) --df.describe() 
    * Statistics for **categorical** attributes (histograms, mode, most/least frequent values, percentage, number of unique values)
        * Histogram of values: E.g., df[<attribute>].value_counts() or seaborn’s distplot()
    * Target statistics
        * Class distribution: E.g., df[<target>].value_counts() or np.bincount(y)
* Multivariate statistics (more than one variable)
    * Correlation, Contingency Tables

### Correlation
* Correlations: How strongly pairs of attributes are related.
* **Scatterplot** matrices visualize attribute-target and attribute-attribute pairwise relationships.
* Correlation matrices measure the linear dependence between features; can be visualized with **heat-maps**

#### Correlation matrix Heat map
`cm = np.corrcoef(df[cols].values.T)`
`ax = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.1f', yticklabels=cols, xticklabels=cols)plt.show()`
<img src="https://i.imgur.com/WAO4Phg.png" height="300" />

Tip: Multi-collinearity
- Highly correlated (pos. or negative) attributes usually degrade performance of linear ML models such as linear and logistic regression models.
- With the regression models, we should **select one of the correlated pairs** and discard the other.
- **Decision Trees** are immune to this problem

## Some commong techniques are:
1. Categorical Encoding - Converting categorical values into numerics, using one-hot encoding.
2. Feature Engineering - Select most relevent features for ML Model.
3. Handling Missing Values - Removing missing or duplicate data. 
nan.() , df.duplicated()

### 1. Categorical Encoding

#### Ordinal Values

Yes --> 1      No --> 0

Small --> 5      Medium --> 10     Large --> 15     None --> 0

condo -->   house -->   Apartment ---> ?? we will use one-hot encoding as if we assign 5,10,15 they are not ordinal in nature

#### One Hot encoding | for Nominal (not ordinal) values | Lib: sklearn

<img src="https://i.imgur.com/HqgHRv8.jpg" width="600" />
<img src="https://mk0analyticsindf35n9.kinstacdn.com/wp-content/uploads/2019/10/2.jpeg" width="600" />

### 2. Handling Missing Values

________________________________
### Text Cleaning | NLtk lib (Natural Language Tool Kit)
<img src="https://i.imgur.com/xADNfIy.png" width="600" />

#### 1 Traform to lower case --> remove white spaces --> Removing Punctiation and Stop words
remove "A", "An", "The", "is", "are"

#### 2 Stemming and Synonym nomalization
jumping --> jump, awesome, wonderful, great --> great

#### 3 Bag of words or N Gram =1
Tokenize each word if N Gram/Unigram is 1 but let say it is 2/Bi-gram or more, it will tokenize words like:
BiGram - "not good", "Very Poor"
TriGram - "hip hip hurrey"

##### 3a Orthogonal Sparse Bigram (OSB)
<img src="https://i.imgur.com/7d3VltX.png" width="600" />

#### 4 TF-idf Term Frequency - Inverse Document Frequency
Used to filter out not importnt common words
<img src="https://i.imgur.com/ZwjMTvz.png" width="800" />

### Use-case of each method
![](https://i.imgur.com/QISoj20.png)

#### 4 Cartesian Product
Create new feature from combination of set of words
<img src="https://i.imgur.com/nCRUlwR.png" width="600" />

#### 5 Date Engineering
Separate Year, Month, Date, Day, Hour, Min, Sec, MilSec

##### Summary of above techniques
<img src="https://i.imgur.com/4nuUtEI.png" width="600" />
________________________________________________________________________________________________

### Numerical Feature Engineering

1. Feature Scaling
2. Binning

#### 1. Feature Scaling
##### Normalization: Scale down the scale of values between 0 and 1, but Outliers can create problems 
so its good to remove any outliers before doing normalization (random-cut forest can be used with outliers).
<img src="https://www.bogotobogo.com/python/scikit-learn/images/scikit-Processing-Datasets-Partitioning-Feature-Selection/X_norm_0.png" height="300" />

##### Standardization: It can also scale down the values with respect to Avg value as 0 and rest values according to Standard deviation, so Outliers does not create much problem.
<img src="/images/standardization.png" height="300" />

#### 2. Binning
Create bins for scattered values like 0-25, 25-50, 50-73
(Gradient descent and kNN requires binning)

### Image Feature Engineering
<img src="https://i.imgur.com/EKNY5Qe.png" width="600" />

## Data Formats supported by Sagemaker

1. File : Load from S3, CSV, JSON, Parquet, png, jpg
2. Pipe : Stream from S3, recordIO-protobuf



____________________________________________

Use fit_transform() only with Training data

Use transform() for Test and Validation

## Data Split
<img src="https://i.imgur.com/zUcDON7.png" height="300" />






