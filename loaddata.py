from ucimlrepo import fetch_ucirepo 
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os
from pathlib import Path
from plotting import graph_with_multiple_x_values
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# metadata 
print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables) 


print(f"Data or Matrix X initially:\n{X}")
print(f"y initially:\n{y}")

# Excluded num, num is the target
feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# could create dict connecting variable names to their specific column of X
mapped_features = {}

# Some of these values have none. We can exclude those. 




graph_with_multiple_x_values(X, y, feature_names, "plots")