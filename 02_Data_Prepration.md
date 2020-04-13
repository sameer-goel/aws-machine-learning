# Data Prepration

Most important and time consuming process of Data Science like Tex cleaning, Missing Values, Outliers, OneHot encoding, Unform data types.

<img src="https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fgilpress%2Ffiles%2F2016%2F03%2FTime-1200x511.jpg" width="600" />

## Most important is to Start with understanding your Data

## Step 1: Descriptive Statistics

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

#### Correlation
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
- **Decision Trees** are immune to this problem.

************************************************************************************

Now we have understood the data, lets work on making this dataset Cleaned
## Step 2: Tyding the data, solving data imbalance to avoid model bias

Down Sampling, Up Sampling, Data Augmentation, Up/down Weights.
<img src="https://i.imgur.com/MsaYUDT.png" height="300" />


## 1. Imputing Missing Values
* Average imputation: Replaces missing values with the **average/mean** value in the column. Useful for **numeric** variables. 
`df['col_name'].fillna((df['col_name'].mean()), inplace=True)`
* Common point imputation: Use the **most common value/mode** for that column to replace missing values. Useful for **categorical** variables.
`df['col_name'].fillna((df['col_name'].mode()), inplace=True)`

#### 1.1 Advance Imputation
We can learn to predict missing values from complete samples using some machine learning techniques. 
* For example: AWS Datawigtool uses Suvervised Learning neural networks to predict missing values in tabular data. https://github.com/awslabs/datawig 

1. MAR Missing at Random
2. MCAR Missing completely at Random
3. MNAR Missing Not at Random

## 2. Feature scaling
- Motivation: Many algorithms are sensitive to features being on different scales, e.g., **gradient descent** and **kNN**
- Solution: Bring features on to the **same scale**.
- Note: Some algorithms like **decision trees** and **random forests** aren’t sensitive to features on different scales
- Common choices (both for linear)
    - Mean/variance standardization
    - MinMaxscaling

### 2.a Normalization/Min-Max scaling: Transoform the between 0 and 1, but Outliers can create problems 
so its good to remove any outliers before doing normalization (random-cut forest can be used with outliers).
<img src="https://i.imgur.com/d3FL118.png" height="200" />

### 2.b Standardization: Scale values to be centered around mean 0 with standard deviation 1, so Outliers does not create much problem.
<img src="https://i.imgur.com/4PIqnfz.png" height="200" />

### 2.c Binning
Create bins for scattered values like 0-25, 25-50, 50-73
(Gradient descent and kNN requires binning)

### 3 Image Feature Engineering
<img src="https://i.imgur.com/EKNY5Qe.png" width="600" />

## Text Cleaning | NLtk lib (Natural Language Tool Kit)

### 1 Traform to lower case --> remove white spaces --> Removing Punctiation and Stop words
remove "A", "An", "The", "is", "are"

### 2 Stemming and Synonym nomalization
jumping --> jump, awesome, wonderful, great --> great

### 3 Bag of words or N Gram =1
Tokenize each word if N Gram/Unigram is 1 but let say it is 2/Bi-gram or more, it will tokenize words like:
BiGram - "not good", "Very Poor"
TriGram - "hip hip hurrey"

#### 3a Orthogonal Sparse Bigram (OSB)
<img src="https://i.imgur.com/7d3VltX.png" width="600" />

### 4 TF-idf Term Frequency - Inverse Document Frequency
Used to filter out not importnt common words
<img src="https://i.imgur.com/ZwjMTvz.png" width="800" />

**Example:** A term frequency–inverse document frequency (tf–idf) matrix using both unigrams and bigrams is built from a text corpus consisting of the following two sentences: { Hello world } and { Hello how are you }. What are the dimensions of the tf–idf vector/matrix? 

**Sol:**There are 2 sentences (or corpus data we are vectorizing) with 5 unique unigrams ('are', 'hello', 'how', 'world', 'you') and there are 4 unique bigrams ('are you', 'hello how', 'hello world', 'how are'). So the vectorized matrix would be (2, 9). 

### Use-case of each method
![](https://i.imgur.com/QISoj20.png)

#### 4 Cartesian Product
Create new feature from combination of set of words
<img src="https://i.imgur.com/nCRUlwR.png" width="600" />

#### 5 Date Engineering
Separate Year, Month, Date, Day, Hour, Min, Sec, MilSec

##### Summary of above techniques
<img src="https://i.imgur.com/4nuUtEI.png" width="600" />

**********************************************************************************************
## Step 3: Feature Selection

Feature engineering:Using domain and data knowledge to create novel featuresas inputs for ML models often more art than science.

Some rules of thumb
- Use intuition: “What information would a humanuse to predict?”
- Try generating many features, then apply dimensionality reduction if needed§Consider transformations of attributes (e.g., squaring)
- Consider combinations of attributes (e.g., multiplication)
- Do not overthink or include too much manual logic
- **scikit-learn:sklearn.feature_extraction**

### PCA: Principle component analysis
Unsupervised learning alogorithm to reduce number of feature but still retain info.


**********************************************************************************************

Now we have Cleaned and Sorted data, clean data is good to read by human but machine models might have to encode those values.
 ## Step 4: Encoding and Data Transformation

## Categorical Encoding

#### Ordinal Values

Yes --> 1      No --> 0

Small --> 5      Medium --> 10     Large --> 15     None --> 0

condo -->   house -->   Apartment ---> ?? we will use one-hot encoding as if we assign 5,10,15 they are not ordinal in nature

Even further for many classes: Averaging the target value for each category. Then, replace the categorical values with the average target value.
<img src="https://i.imgur.com/fQAXVXN.png" height="300" />

#### One Hot encoding | for Nominal (not ordinal) values | Lib: sklearn

<img src="https://i.imgur.com/HqgHRv8.jpg" width="600" />

**********************************************************************
 



