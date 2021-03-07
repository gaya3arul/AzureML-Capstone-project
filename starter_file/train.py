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

def clean_data(data):
    # Dict for cleaning data


    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    Operator = pd.get_dummies(x_df.Operator, prefix="job")
    x_df.drop("Operator", inplace=True, axis=1)
    x_df.drop("Date", inplace=True, axis=1)
    

    y_df = x_df.pop("Failure").apply(lambda s: 1 if s == "Yes" else 0)
    return x_df,y_df

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
datastore_path="https://raw.githubusercontent.com/gaya3arul/nd00333-capstone/master/train.csv"

ds = TabularDatasetFactory.from_delimited_files(path=datastore_path)


x, y = clean_data(ds)

# TODO: Split data into train and test sets.

#dividing X,y into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=403)

### YOUR CODE HERE ###a

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

if __name__ == '__main__':
    main()