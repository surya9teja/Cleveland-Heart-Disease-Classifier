from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv("/home/surya/PycharmProjects/clevland_heart_disease/heart.csv")
print(data.head())
y = data['target']
x = data.loc[:, data.columns != 'target']

# Age distribution
plt.figure(figsize = (8, 8))
sns.distplot(data['age'], color='blue')
plt.title('age distribution', fontsize = 14)
plt.show()

# Uniques value in cp columns
print(data['cp'].value_counts())

# cp Histogram

plt.figure(figsize = (8, 6))
sns.histplot(data['cp'])
plt.title('cp histplot', fontsize = 15)
plt.show()

# trestbps is positive skewd

f, ax = plt.subplots(1,3, figsize=(24, 6))

sns.histplot(data['chol'], ax=ax[0])
ax[0].set_title("Chol histplot")

sns.distplot(data['chol'], ax=ax[1])
ax[1].set_title('Chol distplot')

sns.stripplot(x=data['target'], y=data['chol'], ax=ax[2])
ax[2].set_title('chol vs target')

plt.grid()

plt.show()

# Correlation matrix
corr = data.corr()
p1 = sns.heatmap(corr)
plt.title("Corelation Matrix")
plt.show()

# Old peak

f, ax = plt.subplots(1,3, figsize=(24, 6))

sns.histplot(data['oldpeak'], ax=ax[0])
ax[0].set_title("oldpeak histplot")

sns.distplot(data['oldpeak'], ax=ax[1])
ax[1].set_title('oldpeak distplot')

sns.stripplot(x=data['target'], y=data['oldpeak'], ax=ax[2])
ax[2].set_title('oldpeak vs target')

plt.grid()

plt.show()