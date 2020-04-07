## Data Prepration

Most important and time consuming process of Data Science like Tex cleaning, Missing Values, Outliers, OneHot encoding, Unform data types.

![](https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fgilpress%2Ffiles%2F2016%2F03%2FTime-1200x511.jpg)

#### Some commong techniques are:
1. Categorical Encoding - Converting categorical values into numerics, using one-hot encoding.
2. Feature Engineering - Select most relevent features for ML Model.
3. Handling Missing Values - Removing missing or duplicate data.

### Categorical Encoding

#### Ordinal Values

Yes --> 1      No --> 0

Small --> 5      Medium --> 10     Large --> 15     None --> 0

condo -->   house -->   Apartment ---> ?? we will use one-hot encoding as if we assign 5,10,15 they are not ordinal in nature

#### One Hot encoding | for Nominal (not ordinal) values | Lib: sklearn

![](https://i.imgur.com/HqgHRv8.jpg)
![](https://mk0analyticsindf35n9.kinstacdn.com/wp-content/uploads/2019/10/2.jpeg) 

---
### Text Cleaning
![](https://i.imgur.com/xADNfIy.png)

#### Bag of words or N Gram =1
Tokenize each word if N Gram/Unigram is 1 but let say it is 2/Bi-gram or more, it will tokenize words like:
BiGram - "not good", "Very Poor"
TriGram - "hip hip hurrey"

##### Orthogonal Sparse Bigram (OSB)
![](https://i.imgur.com/7d3VltX.png)

#### TF-idf Term Frequency - Inverse Document Frequency
Used to filter out not importnt common words
![](https://i.imgur.com/ZwjMTvz.png)

### Use-case of each method
![](https://i.imgur.com/QISoj20.png)


