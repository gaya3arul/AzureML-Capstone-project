from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

df = pd.read_csv('./train.csv')

# Preview of the first five rows
print(df.head())

# Explore data
print(df.describe())



# Data columns
df.columns = ['ID', 'Date', 'Temperature', 'Humidity','Operator','Measure1', 'Measure2','Measure3','Measure4','Measure5','Measure6','Measure7','Measure8','Measure9','Measure10','Measure11','Measure12','Measure13','Measure14','Measure15','Hours Since Previous Failure','Failure','?Date.year','?Date.month','?Date.day-of-month','?Date.day-of-week','?Date.hour','?Date.minute','?Date.second']
x = df[['ID', 'Date', 'Temperature', 'Humidity','Operator', 'Measure1', 'Measure2','Measure3','Measure4','Measure5','Measure6','Measure7','Measure8','Measure9','Measure10','Measure11','Measure12','Measure13','Measure14','Measure15','Hours Since Previous Failure','?Date.year','?Date.month','?Date.day-of-month','?Date.day-of-week','?Date.hour','?Date.minute','?Date.second']]
y = df[['Failure']]


# Split data into train and test sets.
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

data = {"train": {"X": x_train, "y": y_train},
        "test": {"X": x_test, "y": y_test}}

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)

    joblib.dump(value=model, filename='outputs/model.pkl')

    
data = pd.read_csv('./train.csv')
if __name__ == '__main__':
    main()