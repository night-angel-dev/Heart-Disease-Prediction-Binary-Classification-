from ucimlrepo import fetch_ucirepo 
from eda_plots import graph_with_multiple_x_values
from preprocessing import make_binary_target
import pandas as pd

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 




# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
y_raw = y.copy()

print("-" * 20)
print("EDA Part: Check original target distribution")
raw_target_counts = (
    y_raw.iloc[:, 0]
    .value_counts(dropna=False)
    .sort_index()
)

print("Original target distribution:")
print(f"{raw_target_counts}\n")


y = make_binary_target(y)
print("-" * 20)
print("EDA Part: Check target distribution after preprocessing")
binary_target_counts = (
    y.iloc[:, 0]
    .value_counts(dropna=False)
    .sort_index()
)

print("Binary target distribution:")
print(f"{binary_target_counts}\n")

print("Binary target percentages:")
print(
    y.iloc[:, 0]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)
print("-" * 20)

print("EDA Part: Checking missing values")
print("Missing values in X:")
print(f"{X.isna().sum()}\n")

print("Missing values in y:")
print(y_raw.isna().sum())

# Include rows that are missing atleast one feature
rows_with_missing_values = X.isna().any(axis=1).sum()
print(f"Rows containing missing feature values: {rows_with_missing_values}")

print("-" * 20)

print("EDA Part: Check data types and non numeric values")
print(f"{X.dtypes}\n")
print("Columns that are not numeric:")
print(X.select_dtypes(exclude="number").columns.tolist())

print("-" * 20)

print("EDA Part: Sumarrize numerical features")
print(X.describe().T)


print("-" * 20)

print("EDA Part: Separate Coninuous and Categorical features")

continuous_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

print(f"Continuous Features:\n{continuous_features}")
print(f"Categorical Features:\n{categorical_features}")

print("-" * 20)

print("EDA Part: Compare Features by target class\n")

# Continuous variables
print(
    X.assign(target=y.iloc[:, 0])
    .groupby("target")[continuous_features]
    .agg(["mean", "median", "std", "min", "max"])
)

# categorical variables
for column in categorical_features:
    print(f"\n{column} by target:")
    print(
        pd.crosstab(
            X[column],
            y.iloc[:, 0],
            normalize="columns"
        ).round(3)
    )

print("-" * 20)
# # metadata 
# print(heart_disease.metadata) 
  
# # variable information 
# print(heart_disease.variables) 


# EDA Part: Record Dataset Shape
print("EDA Part: Record Dataset shape")
print("Feature shape:", X.shape)
print("Target shape:", y_raw.shape)
print("Number of rows:", len(X))
print("Number of features:", X.shape[1])
print("-" * 20)

print("EDA Part: Look for outliers")

for column in continuous_features:
    q1 = X[column].quantile(0.25)
    q3 = X[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = X[
        (X[column] < lower_bound) |
        (X[column] > upper_bound)
    ]

    print(
        f"{column}: {len(outliers)} potential outlier(s), "
        f"range [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

print("-" * 20)


print(f"Data or Matrix X initially:\n{X}")
print(f"Raw y (severity):\n{y_raw}")
print(f"Binary y (classification target):\n{y}")

# Excluded num, num is the target
feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# could create dict connecting variable names to their specific column of X
mapped_features = {}

# Some of these values have none. We can exclude those. 




graph_with_multiple_x_values(X, y, feature_names, "plots")