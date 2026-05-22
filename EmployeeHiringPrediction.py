

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

# DATASET SIZE

n = 1000

# GENERATE NUMERICAL FEATURES

experience = np.random.randint(0, 15, n)

interview_score = np.random.randint(40, 100, n)

projects = np.random.randint(1, 20, n)

study_hours = np.random.randint(1, 10, n)

# GENERATE CATEGORICAL FEATURES

city = np.random.choice(
    ["CANADA", "BERLIN", "CHIAGO"],
    n,
    p=[0.5, 0.3, 0.2]
)

department = np.random.choice(
    ["IT", "HR", "Sales"],
    n
)

# GENERATE CORRELATED FEATURE

salary = (
    experience * 5000
    + projects * 2000
    + interview_score * 1000
    + np.random.normal(0, 10000, n)
)

# CREATE TARGET COLUMN

selected = np.where(
    (
        (experience > 5) &
        (interview_score > 70) &
        (projects > 5)
    ),
    1,
    0
)

# CREATE DATAFRAME

data = pd.DataFrame({
    "experience": experience,
    "interview_score": interview_score,
    "projects": projects,
    "study_hours": study_hours,
    "city": city,
    "department": department,
    "salary": salary,
    "selected": selected
})

# DISPLAY FIRST 5 ROWS

print(data.head())

# CREATE MISSING VALUE MASK

mask = np.random.choice(
    [True, False],
    size=n,
    p=[0.1, 0.9]
)

# INSERT MISSING VALUES

data.loc[mask, "salary"] = np.nan

# CHECK MISSING VALUES

print(data.isna().sum())

# FILL MISSING VALUES

data["salary"] = data["salary"].fillna(
    data["salary"].mean()
)

# ONE HOT ENCODING

data = pd.get_dummies(
    data,
    columns=["city", "department"]
)

# SELECT NUMERICAL COLUMNS

numeric_cols = [
    "experience",
    "interview_score",
    "projects",
    "study_hours",
    "salary"
]

# MIN-MAX NORMALIZATION

data[numeric_cols] = (
    data[numeric_cols] - data[numeric_cols].min()
) / (
    data[numeric_cols].max() - data[numeric_cols].min()
)

# SPLIT FEATURES AND TARGET

X = data.drop("selected", axis=1)

y = data["selected"]

# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# DISPLAY SHAPES

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)