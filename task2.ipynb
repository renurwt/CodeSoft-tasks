# CodSoft Task 2 – Movie Rating Prediction

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (replace this with your actual CSV file if available)
data = pd.DataFrame({
    'Title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy'],
    'Director': ['Dir1', 'Dir2', 'Dir1', 'Dir3', 'Dir2'],
    'Actors': ['Act1,Act2', 'Act3,Act4', 'Act2,Act3', 'Act1,Act5', 'Act4,Act5'],
    'Budget': [100, 30, 75, 50, 45],
    'Rating': [8.2, 6.5, 7.8, 7.0, 6.8]
})

print("Original Dataset:\n", data)

# Drop Title (not used in prediction)
data = data.drop('Title', axis=1)

# Encode categorical columns
le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actors = LabelEncoder()

data['Genre'] = le_genre.fit_transform(data['Genre'])
data['Director'] = le_director.fit_transform(data['Director'])
data['Actors'] = le_actors.fit_transform(data['Actors'])

print("\nEncoded Dataset:\n", data)

# Split features and target
X = data.drop('Rating', axis=1)
y = data['Rating']

# Train-test
