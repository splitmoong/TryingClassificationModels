import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import tree, model_selection
import matplotlib
import math

def calculate_entropy (dataset: pd.DataFrame):
    target = dataset.iloc[:, -1]
    value_counts_dict = target.value_counts("1").to_dict()
    entropy = float(0)
    for vals in value_counts_dict.values():
        entropy = entropy + vals*math.log2(vals)
    entropy = math.fabs(entropy)
    print(entropy)




#makeshift df creation, have to change it cus it has absolute path but fuck it
dataset_df = pd.read_csv("/Users/bhushan/Documents/GitHub/TryingClassificationModels/Dataset/Obesity_Dataset.csv")
X = dataset_df.drop("Class", axis=1)
y = dataset_df["Class"]
#print(X.dtypes)

# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33)
#
# #using scikit decison tree
# dtree = tree.DecisionTreeClassifier()
# dtree.fit(X_train, y_train)
# y_out = dtree.predict(X_test)
# print(y_out)

calculate_entropy(dataset_df)




