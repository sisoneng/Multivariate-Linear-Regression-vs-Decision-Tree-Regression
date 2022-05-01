#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import sklearn
import os

def load_medical_cost_path():
    csv_path = os.path.join("C:\dev\machinelearning\datasets\Medical Cost Personal Datasets", "insurance.csv")
    return pd.read_csv(csv_path)


medcost = load_medical_cost_path()


medcost.hist(bins=50, figsize=(20,15))
plt.show()


### Random Sampling Method
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(medcost, test_size=0.2, random_state=42)
corr_matrix = medcost.corr(method='pearson')
medcost_age_cat1 = medcost_age_cat2 = medcost_age_cat3 = medcost_age_cat4 = 0

for age in medcost["age"]:
    if age >= 0 and age <= 20:
        medcost_age_cat1 = medcost_age_cat1 + 1
    elif age > 20 and age <= 40:
        medcost_age_cat2 = medcost_age_cat2 + 1
    elif age > 40 and age <= 60:
        medcost_age_cat3 = medcost_age_cat3 + 1
    else:
        medcost_age_cat4 = medcost_age_cat4 + 1


test_age_cat1 = test_age_cat2 = test_age_cat3 = test_age_cat4 = 0

for age in test_set["age"]:
    if age >= 0 and age <= 20:
        test_age_cat1 = test_age_cat1 + 1
    elif age > 20 and age <= 40:
        test_age_cat2 = test_age_cat2 + 1
    elif age > 40 and age <= 60:
        test_age_cat3 = test_age_cat3 + 1
    else:
        test_age_cat4 = test_age_cat4 + 1


### Stratified Sampling Method
medcost["age_cat"] = pd.cut(medcost["age"],
bins=[0, 20, 40, 60, np.inf],
labels=[1, 2, 3, 4])

medcost["age_cat"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(medcost, medcost["age_cat"]):
    strat_train_set = medcost.loc[train_index]
    strat_test_set = medcost.loc[test_index]


strat_test_set["age_cat"].value_counts() / len(strat_test_set) 


for set_ in (strat_train_set, strat_test_set):
    set_.drop("age_cat", axis=1, inplace=True)



medcost = strat_train_set.copy()
medcost_nolabels = strat_train_set.drop("charges", axis=1)
medcost_labels = strat_train_set["charges"].copy()
medcost_test_nolabels = strat_test_set.drop("charges", axis=1)
medcost_test_labels = strat_test_set["charges"].copy()


### Age: Dataset vs Test Set using random sampling 


##print(((medcost_age_cat1 / len(medcost)) - (test_age_cat1 / len(test_set))) * 100)
##print(((medcost_age_cat2 / len(medcost)) - (test_age_cat2 / len(test_set))) * 100)
##print(((medcost_age_cat3 / len(medcost)) - (test_age_cat3 / len(test_set))) * 100)
##print(((medcost_age_cat4 / len(medcost)) - (test_age_cat4 / len(test_set))) * 100)


### Age: Dataset vs Test Set using stratified sampling


##print(((medcost_age_cat1 / len(medcost)) - 0.123134) * 100)
##print(((medcost_age_cat2 / len(medcost)) - 0.399254) * 100)
##print(((medcost_age_cat3 / len(medcost)) - 0.410448) * 100)
##print(((medcost_age_cat4 / len(medcost)) - 0.067164) * 100)


corr_matrix = medcost.corr()
corr_matrix["charges"].sort_values(ascending=False)


### Transform categorical features to numerical
numerical_attribs = ['age','bmi', 'children']
categorical_attribs = ["sex", "smoker", "region"]

full_pipeline = ColumnTransformer([
    ("num", StandardScaler(), numerical_attribs),
    ("cat", OneHotEncoder(), categorical_attribs)
])

medcost_prepared = full_pipeline.fit_transform(medcost_nolabels)


lr = LinearRegression()
lr.fit(medcost_prepared, medcost_labels)


some_data = medcost.iloc[:10]
some_data_labels = medcost_labels.iloc[:10]
some_data_prepared = full_pipeline.transform(some_data)
train_set_predictions = lr.predict(some_data_prepared)

##print(train_set_predictions)
##print(list(some_data_labels))



medcost_predictions = lr.predict(medcost_prepared)
mse = mean_squared_error(medcost_labels, medcost_predictions)
lr_rmse = np.sqrt(mse)
print("Linear Regression RMSE without cross-validation: ", lr_rmse)



tree_reg = DecisionTreeRegressor()
tree_reg.fit(medcost_prepared, medcost_labels)



medcost_predictions = tree_reg.predict(medcost_prepared)
mse = mean_squared_error(medcost_labels, medcost_predictions)
tree_rmse = np.sqrt(mse)
print("Decision Tree Regression  RMSE without cross-validation: ", tree_rmse)


### Cross-Validation

lr_scores = cross_val_score(lr, medcost_prepared, medcost_labels,
scoring="neg_mean_squared_error", cv=10)
lr_rmse_scores = np.sqrt(-lr_scores)

tree_scores = cross_val_score(tree_reg, medcost_prepared, medcost_labels,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


print("Linear Regression with Cross-Validation: \n")
display_scores(lr_rmse_scores)

print("\n\n")
print("Decision Tree Regression with Cross-Validation: \n")
display_scores(tree_rmse_scores)



medcost_tr_test_set = full_pipeline.transform(strat_test_set)

print("\n\n")
medcost_predictions = lr.predict(medcost_tr_test_set)
mse = mean_squared_error(medcost_test_labels, medcost_predictions)
lr_rmse = np.sqrt(mse)
print("Linear Regression prediction on test set: ", lr_rmse)

medcost_predictions = tree_reg.predict(medcost_tr_test_set)
mse = mean_squared_error(medcost_test_labels, medcost_predictions)
tree_rmse = np.sqrt(mse)
print("Decision Tree Regression prediction on test set: ", tree_rmse)

print("Conclusion: Linear Regression has lower RMSE on cross-validation than Decision Tree Regression hence, it performs better.")

