
![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202018-03-22%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%206.38.18.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202018-03-22%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%206.38.18.png)



# Titanic: Machine Learning from Disaster


## Contents of the Notebook

### Part1 : Introduction
### Part2 : Load and check data
    1) load data
    
    2) Feature type

    3) Outlier detection
    
    4) Missing values

### Part3 : Feature analysis
    1) Numerical values

    2) Categorical values
    
### Part4 : Filling missing Values
    1) Age

### Part5 : Feature engineering
    1) Name/Title

    2) Family Size

    3) Cabin

    4) Ticket
    
### Part6 : Modeling
    1) Prepare input and test data
    
    2) Model Performance
    
    3) Model choice and submission

## 1. Introduction

I choosed the Titanic competition which is a good way to introduce feature engineering and ensemble modeling. 

This script follows three main parts:

* **Feature analysis**
* **Feature engineering**
* **Modeling**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from scipy import stats
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

sns.set(style='white', context='notebook', palette='deep')
```

## 2. Load and check data
### 2.1 Load data


```python
# Load train and Test set
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
IDtest = test["PassengerId"]       
```


```python
# Check the data set
print("Train data : ", train.shape)
print("Test  data : ", test.shape)
```

    Train data :  (891, 12)
    Test  data :  (418, 11)



```python
# Check the train data set's columns
print("Train data columns Qty :", len(train.columns), "\n\n")
print("Train data columns :", train.columns)
```

    Train data columns Qty : 12 
    
    
    Train data columns : Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')



```python
# states of train data set
# describe the train
train.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# states of train data set
# describe the data by dtype

summary_train = pd.DataFrame()
for col in train.columns:
    
    # column's name
    column_name = col
    
    # check column's type
    dtype = train[column_name].dtype
    
    # check the qty of not null data per each column
    actual_value_qty = len(train.loc[train[column_name].notnull()])
 
    # length of each columns
    rows = len(train[column_name])
    
    # percent of not null values per columns
    actual_value_percent = round((actual_value_qty / rows)*100,1)
    
    # count the unique values per columns
    unique_values = len(train[column_name].unique())
    
    # sum up the null values per columns
    null_qty = train[column_name].isnull().sum()
    
    # make the dataframe
    data = {'column_name' : column_name, 'dtype' : dtype, 'actual_value_qty' : actual_value_qty,'null_qty' : null_qty, \
            'actual_value_percent(%)' : actual_value_percent  ,'unique_values_qty' : unique_values}
    
    summary_train = summary_train.append(data, ignore_index = True)
    

summary_train.pivot_table(index = ['dtype', 'column_name'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>actual_value_percent(%)</th>
      <th>actual_value_qty</th>
      <th>null_qty</th>
      <th>unique_values_qty</th>
    </tr>
    <tr>
      <th>dtype</th>
      <th>column_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">int64</th>
      <th>Parch</th>
      <td>100.0</td>
      <td>891.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>PassengerId</th>
      <td>100.0</td>
      <td>891.0</td>
      <td>0.0</td>
      <td>891.0</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>100.0</td>
      <td>891.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>100.0</td>
      <td>891.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>100.0</td>
      <td>891.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">float64</th>
      <th>Age</th>
      <td>80.1</td>
      <td>714.0</td>
      <td>177.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>100.0</td>
      <td>891.0</td>
      <td>0.0</td>
      <td>248.0</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">object</th>
      <th>Cabin</th>
      <td>22.9</td>
      <td>204.0</td>
      <td>687.0</td>
      <td>148.0</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>99.8</td>
      <td>889.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Name</th>
      <td>100.0</td>
      <td>891.0</td>
      <td>0.0</td>
      <td>891.0</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>100.0</td>
      <td>891.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Ticket</th>
      <td>100.0</td>
      <td>891.0</td>
      <td>0.0</td>
      <td>681.0</td>
    </tr>
  </tbody>
</table>
</div>



Comments :

    - Age, Cabin and Embarked on train data have some missing values. Escpecially, Cabin columns have lot of missing values

### 2-2 Feature type

    1) Seperate Numerical feature and Categorical feature 


```python
# Since Pclass is Categorical feature, I am going to convet it to string in both train and test set
train["Pclass"] = train["Pclass"].astype("str")
test["Pclass"] = test["Pclass"].astype("str")
```


```python
numerical_features = []
categorical_features = []
for f in train.columns:
    if train.dtypes[f] != 'object':
        numerical_features.append(f)
    else:
        categorical_features.append(f)
```


```python
print("Numerical Features Qty :", len(numerical_features),"\n")
print("Numerical Features : ", numerical_features, "\n\n")
print("Categorical Features Qty :", len(categorical_features),"\n")
print("Categorical Features :", categorical_features)
```

    Numerical Features Qty : 6 
    
    Numerical Features :  ['PassengerId', 'Survived', 'Age', 'SibSp', 'Parch', 'Fare'] 
    
    
    Categorical Features Qty : 6 
    
    Categorical Features : ['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']


### 2.3 Outlier detection

    1) By box-and-whisker's IQR   
    
         - The Tukey method (Tukey JW., 1977) to detect ouliers which defines an interquartile range comprised
         between the 1st and 3rd quartile of the distribution values (IQR).


```python
# Outlier detection by Box plot 

def detect_outliers(data, features):
    
    outlier_indices = []
    # iterate over features(columns)
    for feature in features:
        # 1st quartile (25%)
        Q1 = np.percentile(data[feature], 25)
         # 3rd quartile (75%)
        Q3 = np.percentile(data[feature], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
         # outlier step
        outlier_step = 1.5 * IQR
        
        # determine a list of indices of outliers for feature col
        outliers = data[(data[feature] < Q1 - outlier_step) | (data[feature] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outliers)
        
    outlier_indices = Counter(outlier_indices)
    outliers = list( num for num, qty in outlier_indices.items() if qty > 2 )
        
    return outliers   
```

Note : I decided to detect outliers from the numerical values features (Age, SibSp, Sarch and Fare)


```python
# detect outliers from Age, SibSp , Parch and Fare
Outliers_numerical_features = detect_outliers(train,["Age", "SibSp","Parch", "Fare"])

```


```python
train.loc[Outliers_numerical_features]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.00</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>88</th>
      <td>89</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Mabel Helen</td>
      <td>female</td>
      <td>23.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.00</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>159</th>
      <td>160</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Master. Thomas Henry</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>180</th>
      <td>181</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Constance Gladys</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>201</th>
      <td>202</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Mr. Frederick</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>324</th>
      <td>325</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Mr. George John Jr</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>341</th>
      <td>342</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Alice Elizabeth</td>
      <td>female</td>
      <td>24.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.00</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>792</th>
      <td>793</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Stella Anna</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>846</th>
      <td>847</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Mr. Douglas Bullen</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>863</th>
      <td>864</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Miss. Dorothy Edith "Dolly"</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.55</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Comments : 

    - Found 10 outliers and I am going to remove it


```python
# drop outliers
train = train.drop(Outliers_numerical_features, axis = 0).reset_index(drop=True)
```

### 2.4 Missing values

    1) join the train and test set


```python
# in order to handle all missing data 
train_len = len(train)
all_data =  pd.concat([train, test], axis=0).reset_index(drop=True)
```

    2) check for null and missing value


```python
# Fill empty and NaNs values with NaN
all_data = all_data.fillna(np.nan)

# Copy all_data
all_data_cp = all_data.copy()

# check for null values
all_data_null = all_data_cp.isnull().sum()
all_data_null = all_data_null.drop(all_data_null[all_data_null == 0].index).sort_values(ascending=False)

# drop the null values of Survived because Survived missing values correspond to the join testing dataset
del all_data_null['Survived']
```


```python
# make missing dataframe
all_data_missing = pd.DataFrame({'Missing Numbers' :all_data_null})
all_data_null =  all_data_null / len(all_data_cp)*100

# draw the graph for missing data 
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=all_data_null.index, y=all_data_null)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

print("Missing Data Features's Qty : " , all_data_missing.count().values)
print("Total Missing Data's Qty : " , all_data_missing.sum().values)
```

    Missing Data Features's Qty :  [4]
    Total Missing Data's Qty :  [1266]



![png](output_28_1.png)


Comments : 

    - Age and Cabin features have an important part of missing values.

## 3. Feature analysis
### 3.1 Numerical values


```python
# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
ls_numeric = ["Survived","SibSp","Parch","Age","Fare"]
corr = train[ls_numeric].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask = mask, annot=True, fmt = ".2f", cmap = "YlGnBu")
```


![png](output_31_0.png)


Comments :

    - Only Fare feature seems to have a significative correlation with the survival probability.

    - It doesn't mean that the other features are not usefull. Subpopulations in these features can be correlated with the survival. To determine this, we need to explore in detail these features

#### SibSP
Definition : Number of siblings / spouses aboard the Titanic
- Sibling = brother, sister, stepbrother, stepsister
- Spouse = husband, wife (mistresses and fiancés were ignored)


```python
# Explore SibSp feature vs Survived
g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 5 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
```


![png](output_34_0.png)


Comments :

    - It seems that passengers having a lot of siblings/spouses have less chance to survive

    - Single passengers (0 SibSP) or with two other persons (SibSP 1 or 2) have more chance to survive

#### Parch
Definition : Number of parents / children aboard the Titanic
- Parent = mother, father
- Child = daughter, son, stepdaughter, stepson
- Some children travelled only with a nanny, therefore parch=0 for them.


```python
# Explore Parch feature vs Survived
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 5 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
```


![png](output_37_0.png)


Comments :

    - Couple and small families have more chance to survive, more than single (Parch 0),and large families (Parch 5,6 )

#### Age
Definition : The age of the passenger
- Age is fractional if less than 1
- If the age is estimated, is it in the form of xx.5


```python
# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
```


![png](output_40_0.png)


Comments :

    - Age distribution seems to be a tailed distribution, maybe a gaussian distribution.

    - Notice that age distributions are not the same in the survived and not survived subpopulations.
    
    - There is a peak corresponding to young passengers, that have survived. We also see that passengers between 60-80 have less survived. So it seems that very young passengers have more chance to survive.


```python
# Explore Age distibution 
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
```


![png](output_42_0.png)


Comments : 

    - When we superimpose the two densities , we cleary see a peak correponsing (between 0 and 10) to babies and young childrens.

#### Fare
Definition : Passenger fare


```python
# check how many missing values on Fare
all_data["Fare"].isnull().sum()
```




    1




```python
#Fill Fare missing values with the median value
all_data["Fare"] = all_data["Fare"].fillna(all_data["Fare"].median())
```

Comments :

    - Since we have one missing value , i decided to fill it with the median value which will not have an important effect on the prediction.


```python
# Explore Fare distribution 
plt.figure(figsize=(12,5))

plt.subplot(131)
sns.distplot(all_data["Fare"])

plt.subplot(132)
stats.probplot(all_data["Fare"], plot=plt)

plt.subplot(133)
sns.boxplot(all_data["Fare"])
plt.tight_layout()
plt.show()

print("Skewness: %f" % all_data['Fare'].skew())
print("Kurtosis: %f" % all_data['Fare'].kurt())
```


![png](output_48_0.png)


    Skewness: 4.511862
    Kurtosis: 29.183273


Comments : 

    - As we can see, Fare distribution is very skewed. This can lead to overweigth very high values in the model, even if it is scaled. 

    - In this case, it is better to transform it with the log function to reduce this skew. 


```python
all_data["Fare"] = np.log1p(all_data["Fare"])

# Explore Fare distribution 
plt.figure(figsize=(12,5))

plt.subplot(131)
sns.distplot(all_data["Fare"])

plt.subplot(132)
stats.probplot(all_data["Fare"], plot=plt)

plt.subplot(133)
sns.boxplot(all_data["Fare"])
plt.tight_layout()
plt.show()

print("Skewness: %f" % all_data['Fare'].skew())
print("Kurtosis: %f" % all_data['Fare'].kurt())
```


![png](output_50_0.png)


    Skewness: 0.544004
    Kurtosis: 0.921062


Comments : 

    - Skewness is clearly reduced after the log transformation

### 3.2 Categorical values
#### Sex
Definition : Passenger Sex 


```python
g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")
```


![png](output_53_0.png)


Comments :

    - It is clearly obvious that Male have less chance to survive than Female.

    - Sex, might play an important role in the prediction of the survival.

#### Pclass
Definition : A proxy for socio-economic status (SES)

- 1st = Upper   
- 2nd = Middle   
- 3rd = Lower


```python
# Explore Pclass vs Survived
plt.figure(figsize=(15,6))

ax1 = plt.subplot(1,2,1)
sns.barplot(x="Pclass",y="Survived",data=train, palette = "muted", ax=ax1)


# Explore Pclass vs Survived by Sex
ax2 = plt.subplot(1,2,2)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train, palette="muted", ax=ax2)

plt.show()
```


![png](output_56_0.png)


Comments :

    - The passenger survival is not the same in the 3 classes. First class passengers have more chance to survive than second class and third class passengers.

#### Embarked
Definition : Port of Embarkation

- C = Cherbourg 
- Q = Queenstown
- S = Southampton


```python
all_data["Embarked"].isnull().sum()
```




    2



Note : Since we have two missing value , I decided to fill it with the most fequent value of "Embarked"(S)


```python
#Fill Embarked nan values of dataset set with 'S' most frequent value
all_data["Embarked"] = all_data["Embarked"].fillna("S")
```


```python
# Explore Embarked vs Survived 
g = sns.barplot(x="Embarked", y="Survived",  data=train)
```


![png](output_62_0.png)


Comments :

    - It seems that passenger coming from Cherbourg (C) have more chance to survive.

    - My hypothesis is that the proportion of first class passengers is higher for those who came from Cherbourg than Queenstown (Q), Southampton (S).

    - Let's see the Pclass distribution vs Embarked


```python
# Explore Pclass vs Embarked 
g = sns.factorplot("Pclass", col="Embarked",  data=train,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")
```


![png](output_64_0.png)


Comments :

    - Indeed, the third class is the most frequent for passenger coming from Southampton (S) and Queenstown (Q), whereas Cherbourg passengers are mostly in first class which have the highest survival rate.

    - I think that first class passengers were prioritised during the evacuation.

## 4. Filling missing Values
### 4.1 Age


```python
all_data["Age"].isnull().sum()
```




    256



Note :

    - As we see, Age column contains 256 missing values in the whole dataset.

    - Since there is subpopulations that have more chance to survive (children for example), it is preferable to keep the age feature and to impute the missing values. 

    - To adress this problem, I looked at the most correlated features with Age (Sex, Parch , Pclass and SibSP).


```python
# Explore Age vs Sex, Parch , Pclass and SibSP
plt.figure(figsize=(20,10))

ax1 = plt.subplot(2,2,1)
sns.boxplot(y="Age",x="Sex",data=all_data, ax=ax1)

ax2 = plt.subplot(2,2,2)
sns.boxplot(y="Age",x="Sex",hue="Pclass", data=all_data, ax=ax2)

ax3 = plt.subplot(2,2,3)
sns.boxplot(y="Age",x="Parch", data=all_data, ax=ax3)

ax4 = plt.subplot(2,2,4)
sns.boxplot(y="Age",x="SibSp", data=all_data, ax=ax4)

plt.show()
```


![png](output_69_0.png)


Comments :

    - Age distribution seems to be the same in Male and Female subpopulations, so Sex is not informative to predict Age.

    - However, 1st class passengers are older than 2nd class passengers who are also older than 3rd class passengers.

    - Moreover, the more a passenger has parents/children the older he is and the more a passenger has siblings/spouses the younger he is.
    


```python
# convert Sex into categorical value 0 for male and 1 for female
all_data["Sex"] = all_data["Sex"].map({"male": 0, "female":1})
```


```python
numeric = ["Sex","SibSp","Parch","Pclass","Fare","Age"]
corr = all_data[numeric].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask = mask, annot=True, fmt = ".2f", cmap = "YlGnBu")
```


![png](output_72_0.png)


Comments :

     - The correlation map confirms that Age is negatively correlated with Pclass and SibSp.

     - I decided to use SibSP and Pclass in order to impute the missing ages.

     - My plan is to fill Age with the median age of similar rows according to Pclass and SibSp.


```python
# fill Age with the median age of similar rows according to Pclass and SibSp

age_nan = list(all_data["Age"][all_data["Age"].isnull()].index)

for i in age_nan:
    age_median = all_data["Age"].median()
    age_pred = all_data["Age"][((all_data['SibSp'] == all_data.iloc[i]['SibSp']) & (all_data['Pclass'] == all_data.iloc[i]['Pclass']))].median()
    if not np.isnan(age_pred) :
        all_data['Age'].iloc[i] = age_pred
    else :
        all_data['Age'].iloc[i] = age_median
```

## 5. Feature engineering
### 5.1 Name/Title


```python
all_data["Name"].head()
```




    0                              Braund, Mr. Owen Harris
    1    Cumings, Mrs. John Bradley (Florence Briggs Th...
    2                               Heikkinen, Miss. Laina
    3         Futrelle, Mrs. Jacques Heath (Lily May Peel)
    4                             Allen, Mr. William Henry
    Name: Name, dtype: object



Notes :

    - The Name feature contains information on passenger's title.

    - Since some passenger with distingused title may be preferred during the evacuation, it is interesting to add them to the model.


```python
# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in all_data["Name"]]
all_data["Title"] = pd.Series(dataset_title)

g = sns.countplot(x="Title",data=all_data)
g = plt.setp(g.get_xticklabels(), rotation=40) 
```


![png](output_78_0.png)


Comments :

    - There is 17 titles in the dataset, most of them are very rare and we can group them in 4 categories.


```python
# Convert to categorical values Title 
all_data["Title"] = all_data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data["Title"] = all_data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
all_data["Title"] = all_data["Title"].astype(int)
```


```python
g = sns.countplot(all_data["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
```


![png](output_81_0.png)



```python
h = sns.factorplot(x="Title",y="Survived",data=all_data,kind="bar")
h.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
h.set_ylabels("survival probability")
h.fig.set_size_inches(6,5)
```


![png](output_82_0.png)


Commnets :

    - Passengers with Master and rare title have more chance to survive.


```python
# Drop Name variable
all_data.drop(labels = ["Name"], axis = 1, inplace = True)
```


```python
# convert to indicator values Title 
all_data = pd.get_dummies(all_data, columns = ["Title"], prefix = "Title")
```

### 5.2 Family size

Notes : We can imagine that large families will have more difficulties to evacuate, looking for theirs sisters/brothers/parents during the evacuation. So, i choosed to create a "Fize" (family size) feature which is the sum of SibSp , Parch and 1 (including the passenger).


```python
# Create a family size descriptor from SibSp and Parch
all_data["Family_size"] = all_data["SibSp"] + all_data["Parch"] + 1
```


```python
g = sns.barplot(x="Family_size",y="Survived",data = all_data)
```


![png](output_88_0.png)


Comments :

    - The family size seems to play an important role, survival probability is worst for large families.

    - Additionally, i decided to created 4 categories of family size.


```python
# Create new feature of family size
all_data['Single'] = all_data['Family_size'].map(lambda s: 1 if s == 1 else 0)
all_data['Small_F'] = all_data['Family_size'].map(lambda s: 1 if  s == 2  else 0)
all_data['Med_F'] = all_data['Family_size'].map(lambda s: 1 if 3 <= s <= 4 else 0)
all_data['Large_F'] = all_data['Family_size'].map(lambda s: 1 if s >= 5 else 0)

# Drop Family_size variable
all_data.drop(labels = ["Family_size"], axis = 1, inplace = True)
```


```python
plt.figure(figsize=(15,12))

ax1 = plt.subplot(2,2,1)
sns.barplot(x="Single",y="Survived",data=all_data,ax=ax1)

ax2 = plt.subplot(2,2,2)
sns.barplot(x="Small_F",y="Survived",data=all_data,ax=ax2)

ax3 = plt.subplot(2,2,3)
sns.barplot(x="Med_F",y="Survived",data=all_data,ax=ax3)

ax4 = plt.subplot(2,2,4)
sns.barplot(x="Large_F",y="Survived",data=all_data,ax=ax4)

plt.show()
```


![png](output_91_0.png)


Commnets :

    - Barplots of categories show that Small and Medium families have more chance to survive than single passenger and large families.

### 5.3 Cabin 


```python
all_data["Cabin"].head()
```




    0     NaN
    1     C85
    2     NaN
    3    C123
    4     NaN
    Name: Cabin, dtype: object




```python
all_data.Cabin.unique()
```




    array([nan, 'C85', 'C123', 'E46', 'G6', 'C103', 'D56', 'A6', 'B78', 'D33',
           'B30', 'C52', 'B28', 'C83', 'F33', 'F G73', 'E31', 'A5', 'D10 D12',
           'D26', 'C110', 'B58 B60', 'E101', 'F E69', 'D47', 'B86', 'F2', 'C2',
           'E33', 'B19', 'A7', 'C49', 'F4', 'A32', 'B4', 'B80', 'A31', 'D36',
           'D15', 'C93', 'C78', 'D35', 'C87', 'B77', 'E67', 'B94', 'C125',
           'C99', 'C118', 'D7', 'A19', 'B49', 'D', 'C22 C26', 'C106', 'C65',
           'E36', 'C54', 'B57 B59 B63 B66', 'C7', 'E34', 'C32', 'B18', 'C124',
           'C91', 'E40', 'T', 'C128', 'D37', 'B35', 'E50', 'C82', 'B96 B98',
           'E10', 'E44', 'C23 C25 C27', 'A34', 'C104', 'C111', 'C92', 'E38',
           'D21', 'E12', 'E63', 'A14', 'B37', 'C30', 'D20', 'B79', 'E25',
           'D46', 'B73', 'C95', 'B38', 'B39', 'B22', 'C86', 'C70', 'A16',
           'C101', 'C68', 'A10', 'E68', 'B41', 'A20', 'D19', 'D50', 'D9',
           'A23', 'B50', 'A26', 'D48', 'E58', 'C126', 'B71', 'B51 B53 B55',
           'D49', 'B5', 'B20', 'F G63', 'C62 C64', 'E24', 'C90', 'C45', 'E8',
           'B101', 'D45', 'C46', 'D30', 'E121', 'D11', 'E77', 'F38', 'B3',
           'D6', 'B82 B84', 'D17', 'A36', 'B102', 'B69', 'E49', 'C47', 'D28',
           'E17', 'A24', 'C50', 'B42', 'C148', 'B45', 'B36', 'A21', 'D34',
           'A9', 'C31', 'B61', 'C53', 'D43', 'C130', 'C132', 'C55 C57', 'C116',
           'F', 'A29', 'C6', 'C28', 'C51', 'C97', 'D22', 'B10', 'E45', 'E52',
           'A11', 'B11', 'C80', 'C89', 'F E46', 'B26', 'F E57', 'A18', 'E60',
           'E39 E41', 'B52 B54 B56', 'C39', 'B24', 'D40', 'D38', 'C105'], dtype=object)




```python
all_data['Cabin'].isnull().sum()
```




    1007



Notes :

    - The Cabin feature column contains 1007 missing values.

    - I supposed that passengers without a cabin have a missing value displayed instead of the cabin number.


```python
all_data["Cabin"][all_data["Cabin"].notnull()].head()
```




    1      C85
    3     C123
    6      E46
    10      G6
    11    C103
    Name: Cabin, dtype: object




```python
# Replace the Cabin number by the type of cabin 'X' if not
all_data["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in all_data['Cabin'] ])
```

Comments : 

    - I think the Cabin's letter indicates the probable location of the passenger in the Titanic


```python
g = sns.countplot(all_data["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
```


![png](output_101_0.png)



```python
g = sns.factorplot(y="Survived",x="Cabin",data=all_data,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")
```


![png](output_102_0.png)


Comments :

    - We can see that passengers with a cabin have generally more chance to survive than passengers without (X).



```python
# convert to indicator values Cabin 
all_data = pd.get_dummies(all_data, columns = ["Cabin"],prefix="Cabin")
```

### 5.4 Ticket


```python
all_data["Ticket"].head()
```




    0           A/5 21171
    1            PC 17599
    2    STON/O2. 3101282
    3              113803
    4              373450
    Name: Ticket, dtype: object



Notes :

    - It could mean that tickets sharing the same prefixes could be booked for cabins placed together. It could therefore lead to the actual placement of the cabins within the ship.

    - Tickets with same prefixes may have a similar class and survival.

    - So i decided to replace the Ticket feature column by the ticket prefixe. Which may be more informative.


```python
Ticket = []

for i in list(all_data.Ticket):
    if not i.isdigit():
        #Take prefix
        Ticket.append(i.replace(".", "").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append('X')
                      
all_data['Ticket'] = Ticket
all_data['Ticket'].head()

```




    0        A5
    1        PC
    2    STONO2
    3         X
    4         X
    Name: Ticket, dtype: object




```python
all_data = pd.get_dummies(all_data, columns = ["Ticket"], prefix="Ticket")
all_data = pd.get_dummies(all_data, columns = ["Pclass"],prefix="Pclass")
all_data = pd.get_dummies(all_data, columns = ["Embarked"],prefix="Embarked")

# Drop useless variables 
all_data.drop(labels = ["PassengerId"], axis = 1, inplace = True)
```


```python
numerical_features = []
categorical_features = []
for f in all_data.columns:
    if all_data.dtypes[f] != 'object':
        numerical_features.append(f)
    else:
        categorical_features.append(f)
```


```python
categorical_features
```




    []



## 6. Modeling

### 6-1 Prepare input and test data


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
```


```python
all_data = all_data.drop(labels=["Ticket_SCA3", "Ticket_STONOQ", "Ticket_AQ4", "Ticket_AQ3", "Ticket_A", "Ticket_LP"], axis=1)
```


```python
from sklearn.preprocessing import MinMaxScaler

def normalize_age(data):
    scaler = MinMaxScaler()
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
    return data


def normalize_fare(data):
    scaler = MinMaxScaler()
    data["Fare"] = scaler.fit_transform(data["Fare"].values.reshape(-1,1))
    return data

all_data = normalize_age(all_data)
all_data = normalize_fare(all_data)
```


```python
# Separate train dataset and test dataset
train = all_data[:train_len]
X_test = all_data[train_len:]
X_test.drop(labels=["Survived"],axis = 1,inplace=True)
```


```python
# Separate train features and label 
train["Survived"] = train["Survived"].astype(int)
y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"],axis = 1)
y_train = y_train[:,np.newaxis]
```


```python
X_train.shape, y_train.shape
```




    ((881, 59), (881, 1))



### 6-2 Model performance

### 6-2-1 DecisionTree
 - A decision tree is a classification model that divides the independent variable space by applying various rules sequentially. Can be used for both classification and regression

- The way to determine the rule is to find the best independent variable and reference value that makes the entropy between the parent node and child node the lowest. The quantification of these criteria is the information gain


- The information gain is a value indicating how much the entropy of the random variable Y is reduced by the condition of X. It is defined as the entropy of Y minus the conditional entropy of Y for X
$$
IG[Y,X] = H[Y] - H[Y|X]
$$



```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Setup the K-Fold
k_fold = KFold(n_splits=10, shuffle=True, random_state=2018)

# Score list
scores = []

# max_depth_list
max_depth_list =[]
diff_list = []

pre_score = 0

# Change max_depth to check cross_varidation score
for max_depth in range(1,20):
    globals()['tree%s' % max_depth] = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_leaf=5).fit(X_train, y_train)
    score = cross_val_score(globals()['tree%s' % max_depth],X_train, y_train, cv=k_fold, n_jobs=-1, scoring="accuracy").mean()
    print('{}_cross_val_scroe:{}'.format('tree%s' % max_depth, score))
    scores.append(score)
    max_depth_list.append(max_depth)
    diff = score - pre_score 
    pre_score = score
    diff_list.append(diff)

plt.figure(figsize=(17,6))

plt.subplot(121)
plt.bar(max_depth_list[1:-1], diff_list[1:-1])
plt.ylabel('CV score fluctuation')
plt.xlabel('Max_depth')
plt.axhline(np.max(diff_list[1:-1]), linestyle=':', c = 'r', linewidth=4)

plt.subplot(122)
plt.bar(max_depth_list, scores)
plt.ylabel('CV score')
plt.xlabel('Max_depth')
diff_list_max_index = np.argmax(diff_list[1:-1]) + 2
plt.axvline(diff_list_max_index, linestyle=':', c = 'r', linewidth=4)

plt.show()
```

    tree1_cross_val_scroe:0.7843207354443309
    tree2_cross_val_scroe:0.7820863125638406
    tree3_cross_val_scroe:0.8195097037793667
    tree4_cross_val_scroe:0.8138406537282943
    tree5_cross_val_scroe:0.8172369765066394
    tree6_cross_val_scroe:0.8149642492339121
    tree7_cross_val_scroe:0.8183861082737487
    tree8_cross_val_scroe:0.820658835546476
    tree9_cross_val_scroe:0.8138661899897854
    tree10_cross_val_scroe:0.8195224719101123
    tree11_cross_val_scroe:0.8240679264555668
    tree12_cross_val_scroe:0.8206843718079672
    tree13_cross_val_scroe:0.8161389172625126
    tree14_cross_val_scroe:0.8150025536261492
    tree15_cross_val_scroe:0.8138661899897854
    tree16_cross_val_scroe:0.8229570990806947
    tree17_cross_val_scroe:0.8161389172625129
    tree18_cross_val_scroe:0.81841164453524
    tree19_cross_val_scroe:0.8150025536261492



![png](output_122_1.png)



```python
# The performance is best when max_depth is 3
model_1 = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5).fit(X_train, y_train)
print("Score : {}".format(scores[diff_list_max_index-1]))
```

    Score : 0.8195097037793667



```python
from sklearn.metrics import *
print("Confusion Matrix : \n", confusion_matrix(y_train, model_1.predict(X_train)), "\n\n")
print("10-fold Cross Validation Report: \n",classification_report(y_train, model_1.predict(X_train)))
```

    Confusion Matrix : 
     [[484  57]
     [ 93 247]] 
    
    
    10-fold Cross Validation Report: 
                  precision    recall  f1-score   support
    
              0       0.84      0.89      0.87       541
              1       0.81      0.73      0.77       340
    
    avg / total       0.83      0.83      0.83       881
    


### 6-2-2 Randomforest

 - Random Forest is a model combining method that uses Decision Tree as an individual model.
 
 
 - At the time of node separation, the independent variable dimension is reduced randomly and then independent variable is selected. This reduces the variability of model performance because the correlation between individual models is reduced
 


```python
from sklearn.ensemble import RandomForestClassifier

# Score list
scores = []

# max_depth_list
max_depth_list =[]
diff_list = []
pre_score = 0

for max_depth in range(1,20):
    globals()['tree%s' % max_depth] = RandomForestClassifier(max_depth=max_depth, n_estimators=10).fit(X_train, y_train)
    score = cross_val_score(globals()['tree%s' % max_depth],X_train, y_train, cv=k_fold, n_jobs=-1, scoring="accuracy").mean()
    print('{}_cross_val_scroe:{}'.format('tree%s' % max_depth, score))
    scores.append(score)
    max_depth_list.append(max_depth)
    diff = score - pre_score 
    pre_score = score
    diff_list.append(diff)

plt.figure(figsize=(17,6))

plt.subplot(121)
plt.bar(max_depth_list[1:-1], diff_list[1:-1])
plt.ylabel('CV score fluctuation')
plt.xlabel('Max_depth')
plt.axhline(np.max(diff_list[1:-1]), linestyle=':', c = 'r', linewidth=4)

plt.subplot(122)
plt.bar(max_depth_list, scores)
plt.ylabel('CV score')
plt.xlabel('Max_depth')
diff_list_max_index = np.argmax(diff_list[1:-1]) + 2
plt.axvline(diff_list_max_index, linestyle=':', c = 'r', linewidth=4)

plt.show() 
```

    tree1_cross_val_scroe:0.7423518896833504
    tree2_cross_val_scroe:0.796782431052094
    tree3_cross_val_scroe:0.8206332992849846
    tree4_cross_val_scroe:0.8263023493360573
    tree5_cross_val_scroe:0.8240551583248212
    tree6_cross_val_scroe:0.8059371807967313
    tree7_cross_val_scroe:0.8138278855975486
    tree8_cross_val_scroe:0.8161006128702757
    tree9_cross_val_scroe:0.8172114402451482
    tree10_cross_val_scroe:0.8161644535240041
    tree11_cross_val_scroe:0.8161133810010215
    tree12_cross_val_scroe:0.8138534218590399
    tree13_cross_val_scroe:0.8160750766087845
    tree14_cross_val_scroe:0.7967951991828397
    tree15_cross_val_scroe:0.8149642492339122
    tree16_cross_val_scroe:0.819496935648621
    tree17_cross_val_scroe:0.8047369765066394
    tree18_cross_val_scroe:0.801302349336057
    tree19_cross_val_scroe:0.805873340143003



![png](output_127_1.png)



```python
# The performance is best when max_depth is 2
model_2 = RandomForestClassifier(max_depth=2, n_estimators=10).fit(X_train, y_train)
print("Score : {}".format(scores[diff_list_max_index-1]))
```

    Score : 0.796782431052094



```python
from sklearn.metrics import *
print("Confusion Matrix : \n", confusion_matrix(y_train, model_2.predict(X_train)), "\n\n")
print("10-fold Cross Validation Report: \n",classification_report(y_train, model_2.predict(X_train)))
```

    Confusion Matrix : 
     [[463  78]
     [108 232]] 
    
    
    10-fold Cross Validation Report: 
                  precision    recall  f1-score   support
    
              0       0.81      0.86      0.83       541
              1       0.75      0.68      0.71       340
    
    avg / total       0.79      0.79      0.79       881
    


### 6-2-3 AdaBoost

- AdaBoost can be used in conjunction with many other types of learning algorithms to improve performance. The output of the other learning algorithms ('weak learners') is combined into a weighted sum that represents the final output of the boosted classifier
 
 
- AdaBoost is adaptive in the sense that subsequent weak learners are tweaked in favor of those instances misclassified by previous classifiers. 


```python
from sklearn.ensemble import AdaBoostClassifier

# Score list
scores = []

# max_depth_list
max_depth_list =[]
diff_list = []
pre_score = 0

for max_depth in range(1,20):
    globals()['tree%s' % max_depth] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth,random_state=0),n_estimators=100, algorithm="SAMME").fit(X_train, y_train)
    score = cross_val_score(globals()['tree%s' % max_depth],X_train, y_train, cv=k_fold, n_jobs=-1, scoring="accuracy").mean()
    print('{}_cross_val_scroe:{}'.format('tree%s' % max_depth, score))
    scores.append(score)
    max_depth_list.append(max_depth)
    diff = score - pre_score 
    pre_score = score
    diff_list.append(diff)

plt.figure(figsize=(17,6))

plt.subplot(121)
plt.bar(max_depth_list[1:-1], diff_list[1:-1])
plt.ylabel('CV score fluctuation')
plt.xlabel('Max_depth')
plt.axhline(np.max(diff_list[1:-1]), linestyle=':', c = 'r', linewidth=4)

plt.subplot(122)
plt.bar(max_depth_list, scores)
plt.ylabel('CV score')
plt.xlabel('Max_depth')
diff_list_max_index = np.argmax(diff_list[1:-1]) + 2
plt.axvline(diff_list_max_index, linestyle=':', c = 'r', linewidth=4)

plt.show() 
```

    tree1_cross_val_scroe:0.8228804902962207
    tree2_cross_val_scroe:0.8263023493360573
    tree3_cross_val_scroe:0.8229570990806947
    tree4_cross_val_scroe:0.8229315628192031
    tree5_cross_val_scroe:0.8206716036772217
    tree6_cross_val_scroe:0.8036006128702757
    tree7_cross_val_scroe:0.8081460674157304
    tree8_cross_val_scroe:0.8002170582226762
    tree9_cross_val_scroe:0.811567926455567
    tree10_cross_val_scroe:0.8127042900919305
    tree11_cross_val_scroe:0.8013661899897853
    tree12_cross_val_scroe:0.8070224719101123
    tree13_cross_val_scroe:0.8058861082737486
    tree14_cross_val_scroe:0.8059244126659857
    tree15_cross_val_scroe:0.807035240040858
    tree16_cross_val_scroe:0.7933988764044944
    tree17_cross_val_scroe:0.7945480081716035
    tree18_cross_val_scroe:0.8002170582226762
    tree19_cross_val_scroe:0.8036133810010213



![png](output_132_1.png)



```python
# The performance is best when max_depth is 10
model_3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10,random_state=0),n_estimators=100, algorithm="SAMME").fit(X_train, y_train)
print("Score : {}".format(scores[diff_list_max_index-1]))
```

    Score : 0.811567926455567



```python
from sklearn.metrics import *
print("Confusion Matrix : \n", confusion_matrix(y_train, model_3.predict(X_train)), "\n\n")
print("10-fold Cross Validation Report: \n",classification_report(y_train, model_3.predict(X_train)))
```

    Confusion Matrix : 
     [[539   2]
     [  5 335]] 
    
    
    10-fold Cross Validation Report: 
                  precision    recall  f1-score   support
    
              0       0.99      1.00      0.99       541
              1       0.99      0.99      0.99       340
    
    avg / total       0.99      0.99      0.99       881
    


### 6-2-4 Support Vector Machine

- Support vector machine is supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis

![image.png](attachment:image.png)


```python
from sklearn.svm import SVC
model_4 = SVC(kernel='linear').fit(X_train, y_train)
```


```python
score = cross_val_score(model_4,X_train, y_train, cv=k_fold, n_jobs=-1, scoring="accuracy").mean()
print("Score : {}".format(score))
```

    Score : 0.8161006128702757



```python
from sklearn.metrics import *
print("Confusion Matrix : \n", confusion_matrix(y_train, model_4.predict(X_train)), "\n\n")
print("10-fold Cross Validation Report: \n",classification_report(y_train, model_4.predict(X_train)))
```

    Confusion Matrix : 
     [[478  63]
     [ 83 257]] 
    
    
    10-fold Cross Validation Report: 
                  precision    recall  f1-score   support
    
              0       0.85      0.88      0.87       541
              1       0.80      0.76      0.78       340
    
    avg / total       0.83      0.83      0.83       881
    


### 6-2-5 Naive Bayes classification

- Naive Bayes classifiers are a family of simple "probabilistic classifiers "based on applying Bayes' theorem with strong (naive) independence assumptions between the features


- Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem. Maximum-likelihood training can be done by evaluating a closed-form expression which takes linear time, rather than by expensive iterative approximation as used for many other types of classifiers.


- In the Naive Bayes classification model, the probability of combining x vectors is the product of the probability of individual elements $x_i$

$$
P(x_1, \ldots, x_n \mid y = C_k) = \prod_{i=1}^n P(x_i \mid y = C_k)
$$
$$
P(y = C_k \mid x) \;\; \propto \;\; \prod_{i=1}^n P(x_{i} \mid y = C_k)\; P(y = C_k)
$$


```python
from sklearn.naive_bayes import BernoulliNB
model_5 = BernoulliNB().fit(X_train, y_train)
```


```python
score = cross_val_score(model_5,X_train, y_train, cv=k_fold, n_jobs=-1, scoring="accuracy").mean()
print("Cross Score : {}".format(score))
```

    Cross Score : 0.7956716036772217



```python
from sklearn.metrics import *
print("Confusion Matrix : \n", confusion_matrix(y_train, model_5.predict(X_train)), "\n\n")
print("10-fold Cross Validation Report: \n",classification_report(y_train, model_5.predict(X_train)))
```

    Confusion Matrix : 
     [[431 110]
     [ 63 277]] 
    
    
    10-fold Cross Validation Report: 
                  precision    recall  f1-score   support
    
              0       0.87      0.80      0.83       541
              1       0.72      0.81      0.76       340
    
    avg / total       0.81      0.80      0.81       881
    


### 6-2-6 VotingClassifier

- VotingCalssifier is an ensemble subpackage of the Scikit-Learn class for the majority method.


- The majority method is divided into two categories, Hard Voting and Soft Voting.

 * Hard voting: Simple voting. Results of individual models
 * Soft voting: The sum of the conditional probabilities of the individual models 


```python
from sklearn.ensemble import VotingClassifier
model_6 = VotingClassifier(estimators=[('DC', model_1), ('RF', model_2),('AD', model_3),('SVM', model_4),('NB', model_5)], 
                            voting='hard').fit(X_train, y_train)

```


```python
score = cross_val_score(model_6,X_train, y_train, cv=k_fold, n_jobs=-1, scoring="accuracy").mean()
print("Score : {}".format(score))
```

    Score : 0.8308605720122575



```python
print("Confusion Matrix : \n", confusion_matrix(y_train, model_6.predict(X_train)), "\n\n")

print("10-fold Cross Validation Report: \n",classification_report(y_train, model_6.predict(X_train)))
```

    Confusion Matrix : 
     [[484  57]
     [ 80 260]] 
    
    
    10-fold Cross Validation Report: 
                  precision    recall  f1-score   support
    
              0       0.86      0.89      0.88       541
              1       0.82      0.76      0.79       340
    
    avg / total       0.84      0.84      0.84       881
    


### 6-3 Model choice and submission

### 6-3-1 Model : VotingClassifier


```python
# Predict
pred = model_6.predict(X_test)
```


```python
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.distplot(y_train,ax=ax1,bins=50)
sns.distplot(pred,ax=ax2,bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a197c6208>




![png](output_153_1.png)


### 6-3-2 Submission


```python
submission = pd.read_csv("./Submit/gender_submission.csv")
submission

submission["Survived"] = pred


print(submission.shape)
submission.head()
```

    (418, 2)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Export csv file
submission.to_csv("../2_Titanic_Machine_Learning_from_Disaster/Submit/submission_"+str(score)+".csv".format(score), index=False)
```
