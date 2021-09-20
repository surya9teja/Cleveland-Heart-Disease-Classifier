# Cleveland Heart Disease Binary Classifier

A binary classifier build using PyTorch using the Kaggle dataset Cleveland Heart disease.

## Required Python Modules

- [pytorch-1.9](https://github.com/pytorch/pytorch)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Pandas-1.3.2](https://pandas.pydata.org/)

The dataset consists of 303 individuals data. There are 14 columns in the dataset, which are described below.

1.  ***Age***: displays the age of the individual.
2.  ***Sex***: displays the gender of the individual using the following format :
    1 = male
    0 = female
3.  ***Chest-pain type***: displays the type of chest-pain experienced by the individual using the following format :
    1 = typical angina
    2 = atypical angina
    3 = non — anginal pain
    4 = asymptotic
4.  ***Resting Blood Pressure***: displays the resting blood pressure value of an individual in mmHg (unit)
5.  ***Serum Cholestrol***: displays the serum cholesterol in mg/dl (unit)
6.  ***Fasting Blood Sugar***: compares the fasting blood sugar value of an individual with 120mg/dl.
    If fasting blood sugar > 120mg/dl then : 1 (true)
    else : 0 (false)
7.  ***Resting ECG*** : displays resting electrocardiographic results
    0 = normal
    1 = having ST-T wave abnormality
    2 = left ventricular hyperthrophy
8.  ***Max heart rate achieved*** : displays the max heart rate achieved by an individual.
9.  ***Exercise induced angina*** :
    1 = yes
    0 = no
10. ***ST depression induced by exercise relative to rest***: displays the value which is an integer or float.
11. ***Peak exercise ST segment*** :
    1 = upsloping
    2 = flat
    3 = downsloping
12. ***Number of major vessels (0–3) colored by flourosopy*** : displays the value as integer or float.
13. ***Thal*** : displays the thalassemia :
    3 = normal
    6 = fixed defect
    7 = reversible defect
14. ***Diagnosis of heart disease*** : Displays whether the individual is suffering from heart disease or not :
    0 = absence
    1, 2, 3, 4 = present.

[source](https://towardsdatascience.com/heart-disease-prediction-73468d630cfc)

## Data Analysis

### Age Distribution

Here, target = 1 implies that the person is suffering from heart disease and target = 0 implies the person is not suffering.

![Age Distribution](https://github.com/surya9teja/cleveland_heart_disease/blob/master/Assets/age_distribution.png)

### Correlation Matrix

![Correlation Matrix](https://github.com/surya9teja/cleveland_heart_disease/blob/master/Assets/Correlation_Matrix.png)

### CP Histogram
![CP Hist](https://github.com/surya9teja/cleveland_heart_disease/blob/master/Assets/CP_Histogram.png)

### Old Peak Skew

![old peak skew](https://github.com/surya9teja/cleveland_heart_disease/blob/master/Assets/old_peak.png)

### Trest Skew

![Trest Skew](https://github.com/surya9teja/cleveland_heart_disease/blob/master/Assets/trest_skewd.png)
