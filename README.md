# Ex-06-Feature-Transformation
# Aim:
1.To read and perform feature transformation for the given dataset.

# Explanation:
Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column (feature) and transform the values, which are useful for our further analysis. It is a technique by which we can boost our model performance.

# Algorithm:
# STEP 1
Read the given Data

# STEP 2
Clean the Data Set using Data Cleaning Process

# STEP 3
Apply Feature Transformation techniques to all the features of the data set

# STEP 4
Save the data to the file

# Program:

Name : Ashika Jubi R 


Register numnber : 212221040020

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

```


# Output:
# Dataset:

![GITHUB](d.png)

# Head:
![GITHUB](d1.png)

# Null data:
![GITHUB](d2.png)

# Information:
![GITHUB](d3.png)

# Description:
![GITHUB](d4.png)

# Highly Positive Skew:
![GITHUB](d5.png)

# Highly Negative Skew:
![GITHUB](d6.png)

# Moderate Positive Skew:
![GITHUB](d7.png)

# Moderate Negative Skew:
![GITHUB](d8.png)

# Log of Highly Positive Skew:
![GITHUB](d9.png)

# Log of Moderate Positive Skew:
![GITHUB](d10.png)

# Reciprocal of Highly Positive Skew:
![GITHUB](d11.png)

# Square root tranformation:
![GITHUB](d12.png)

# Power transformation of Moderate Positive Skew:
![GITHUB](d13.png)

# Power transformation of Moderate Negative Skew:
![GITHUB](d14.png)

# Quantile transformation:
![GITHUB](d15.png)

# Result:
Thus, Feature transformation is performed and executed successfully for the given dataset.


