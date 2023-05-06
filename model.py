import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# Load data from CSV file
url = 'https://raw.githubusercontent.com/samiha26/UMHACK2023/main/FilteredDataset.csv'
df = pd.read_csv(url)

# Preview the first 5 rows of the dataset
print(df.head())

#filtering
#converting object types to int
df['revenue_growh'] = df['revenue_growh'].str.rstrip('%').astype('float') / 100.0
df['employee_growth_6'] = df['employee_growth_6'].str.rstrip('%').astype('float') / 100.0
df['employee_growth_12'] = df['employee_growth_12'].str.rstrip('%').astype('float') / 100.0
# df['total_funding_c'] = df['total_funding_c'].astype('float')

# Create target variable
# df['is_unicorn'] = (df['revenue_c'] >= 10000000).astype(int)
# Create a new column called 'is_unicorn'
df.loc[(df['EBIT_c'] >= 100000) & (df['total_funding_c'] >= 1000000) & (df['revenue_c'] >= 1000000), 'is_unicorn'] = 1
df.loc[df['is_unicorn'].isnull(), 'is_unicorn'] = 0


# Select relevant features and target variable
X = df[['total_funding_c', 'EBIT_c', 'last_round_size_c', 'revenue_c', 'revenue_growh', 'employee_growth_6', 'employee_growth_12', 'num_funding_rounds', 'num_shareholders']]
y = df['is_unicorn']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=52)

# Fit the random forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)
print("y-pred:", y_pred)
#rf.predict(test[x])

# Compute accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(rf,"rf.joblib")