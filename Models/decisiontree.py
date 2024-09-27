import pandas as pd
import numpy as np
from sklearn import tree

dataset_df = pd.read_csv("/Users/bhushan/Documents/GitHub/TryingClassificationModels/Dataset/Obesity_Dataset.csv")
print(dataset_df.dtypes)