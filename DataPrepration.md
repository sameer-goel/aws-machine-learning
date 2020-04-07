## Data Prepration

Most important and time consuming process of Data Science like Tex cleaning, Missing Values, Outliers, OneHot encoding, Unform data types.

<img src="https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fgilpress%2Ffiles%2F2016%2F03%2FTime-1200x511.jpg" width="600" />

#### Some commong techniques are:
1. Categorical Encoding - Converting categorical values into numerics, using one-hot encoding.
2. Feature Engineering - Select most relevent features for ML Model.
3. Handling Missing Values - Removing missing or duplicate data. 
nan.() , df.duplicated()

### Categorical Encoding

#### Ordinal Values

Yes --> 1      No --> 0

Small --> 5      Medium --> 10     Large --> 15     None --> 0

condo -->   house -->   Apartment ---> ?? we will use one-hot encoding as if we assign 5,10,15 they are not ordinal in nature

#### One Hot encoding | for Nominal (not ordinal) values | Lib: sklearn

<img src="https://i.imgur.com/HqgHRv8.jpg" width="600" />
<img src="https://mk0analyticsindf35n9.kinstacdn.com/wp-content/uploads/2019/10/2.jpeg" width="600" /> 

________________________________
### Text Cleaning
<img src="https://i.imgur.com/xADNfIy.png" width="600" />

#### 1 Removing Punctiation and traform to lower case
remove "A", "An", "The", "is", "are"

#### 2 Bag of words or N Gram =1
Tokenize each word if N Gram/Unigram is 1 but let say it is 2/Bi-gram or more, it will tokenize words like:
BiGram - "not good", "Very Poor"
TriGram - "hip hip hurrey"

##### 1a Orthogonal Sparse Bigram (OSB)
<img src="https://i.imgur.com/7d3VltX.png" width="600" />

#### 3 TF-idf Term Frequency - Inverse Document Frequency
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
##### Standardization: It can also scale down the values with respect to Avg value as 0 and rest values according to Standard deviation, so Outliers does not create much problem.
<img src="/images/standardization.png" height="300" />

#### 2. Binning
Create bins for scattered values like 0-25, 25-50, 50-73
(Gradient descent and kNN requires binning)

### Image Feature Engineering
<img src="https://i.imgur.com/EKNY5Qe.png" width="600" />




Use fit_transform() only with Training data

Use transform() for Test and Validation

