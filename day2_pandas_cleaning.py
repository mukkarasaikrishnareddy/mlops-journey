import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#seed data
np.random.randint(42)

#data size
n = 1000


experience = np.random.randint(0,15,n)
interview_score = np.random.randint(40,100,n)
projects = np.random.randint(0,10,n)
study_hours = np.random.randint(1,10,n)

department = np.random.choice(['HR','IT','Finance','Marketing','Sales'], n)
city = np.random.choice(['New York','Los Angeles','Chicago','Houston','Phoenix'], n,p=[0.3,0.25,0.2,0.15,0.1])
education_level = np.random.choice(['High School','Bachelor','Master','PhD'], n,p=[0.2,0.5,0.25,0.05])

salary = (experience *5000 +interview_score *2000 +projects *1000 +np.random.normal(0,10000,n))
selected = np.where((experience > 5) & (interview_score >70) & (projects > 5),1,0)
data = pd.DataFrame({
    "experience": experience,
    "interview_score": interview_score,
    "projects": projects,
    "study_hours": study_hours,
    "department": department,
    "city": city,
    "education_level": education_level,
    "salary": salary,
    "selected": selected
})
print(data.head())
print(data.tail())
print(data.shape)
print(data.info())
print(data.describe())
print(data["department"].value_counts())
print(data["department"].unique())
mask = np.random.choice([True, False], size= n, p=[0.1, 0.9])
data.loc[mask,"salary"] = np.nan
print(data.isna().sum())
median_salary = data["salary"].median()
data["salary"] = data["salary"].fillna(median_salary)
data = pd.concat(
    [data,data.iloc[:5]],ignore_index = True)
print(data.duplicated().sum())
data = data.drop_duplicates()
data["senior"] =np.where(data["experience"]>8,1,0)
data["high_salary"] = np.where(data["salary"]>data["salary"].mean(),1,0)
numeric_cols = [
    "experience",
    "interview_score",
    "projects",
    "study_hours",
    "salary"
]
data[numeric_cols] = (
    data[numeric_cols] - data[numeric_cols].min()
) / (
    data[numeric_cols].max() - data[numeric_cols].min()
)
data = pd.get_dummies(
    data,
    columns=[
        "department",
        "city",
        "education_level"
    ]
)
plt.hist(
    data["salary"],
    bins=20
)

plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Salary Distribution")

plt.show()
plt.scatter(
    data["experience"],
    data["salary"]
)

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")

plt.show()
plt.boxplot(data["salary"])

plt.title("Salary Boxplot")

plt.show()
data.filter(like="department_").sum().plot(
    kind="bar"
)

plt.title("Department Counts")

plt.show()
corr = data.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True
)

plt.title("Correlation Heatmap")

plt.show()

print(
    data.sort_values(
        by="salary",
        ascending=False
    ).head()
)
print(
    data.groupby("selected")["experience"].mean()
)
print(
    data.groupby("selected")["salary"].agg(
        ["mean", "max", "min"]
    )
)
X = data.drop("selected", axis=1)

y = data["selected"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)