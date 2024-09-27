import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("husl", 7)
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/content/heart.csv")
df

Description of each feature:

age - age in years

sex - sex (1 = male; 0 = female)

cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 0 = asymptomatic)

trestbps - resting blood pressure (in mm Hg on admission to the hospital)

chol - serum cholestoral in mg/dl

fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)

restecg - resting electrocardiographic results (1 = normal; 2 = having ST-T wave abnormality; 0 = hypertrophy)

thalach - maximum heart rate achieved

exang - exercise induced angina (1 = yes; 0 = no)

oldpeak - ST depression induced by exercise relative to rest

slope - the slope of the peak exercise ST segment (2 = upsloping; 1 = flat; 0 = downsloping)

ca - number of major vessels (0-3) colored by flourosopy

thal - 2 = normal; 1 = fixed defect; 3 = reversable defect

num - the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < diameter narrowing; Value 1 = > 50% diameter narrowing)



df.head()
df.describe()
df[df.duplicated(keep = False)]

df_cleaned = df.drop_duplicates()
df_cleaned.duplicated().sum()
df_cleaned.nunique().sort_values()

correlation_matrix= df_cleaned.corr()
correlation_matrix['output']


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


Variables_to_encode = ['cp', 'restecg', 'slp', 'caa', 'thall']

df_encoded = pd.get_dummies(df_cleaned, columns = Variables_to_encode, drop_first = True)
