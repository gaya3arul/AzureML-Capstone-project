
#Predict survival of patients with heart failure using AutoML and HyperDrive

## Table of contents
   * [Overview](#Overview)
   * [Project Set Up and Installation](#Project-Set-Up-and-Installation)
   * [Dataset](#Dataset)
   * [Automated ML](#Automated-ML)
   * [Hyperparameter Tuning](#Hyperparameter-Tuning)
   * [Model Deployment](#Model-Deployment)
   * [Screen Recording](#Screen-Recording)
   * [Comments and future improvements](#Comments-and-future-improvements)
   * [Dataset Citation](#Dataset-Citation)
   * [References](#References)

***

## Overview
The current project uses machine learning to predict patients’ survival based on their medical data. The dataset is used from kaggle . 
Dataset: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv

As part of this capstone project, after registering the dataset into Azure Machine Learning Studio, I need to create two experiments 

i) To train a Model using Logistic Regression Model using custom scikit-learn and tune the hyperparameters using HyperDrive.
ii) To train a Model using Automated ML 

Given below is the workflow for this project
![Project Workflow](https://github.com/gaya3arul/nd00333-capstone/blob/master/starter_file/screenshots-capstone/proj-workflow.png) 

## Project Set Up and Installation
In order to run the project in Azure Machine Learning Studio, we will need the two Jupyter Notebooks:

- `hyperparameter_tuning.ipynb`: for the HyperDrive experiment.
- `automl.ipynb`: for the AutoML experiment;


The following files are also necessary:

- `heart_failure_clinical_records_dataset.csv`: the dataset file.
- `train.py`: a basic script for manipulating the data used in the HyperDrive experiment;
- `scoring_file_v_1_0_0.py`: the script used to deploy the model which is downloaded from within Azure Machine Learning Studio; &
- `env.yml`: the environment file which is also downloaded from within Azure Machine Learning Studio.


## Dataset

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

The dataset contains 13 features:

| Feature | Explanation | Measurement |
| :---: | :---: | :---: |
| *age* | Age of patient | Years (40-95) |
| *anaemia* | Decrease of red blood cells or hemoglobin | Boolean (0=No, 1=Yes) |
| *creatinine-phosphokinase* | Level of the CPK enzyme in the blood | mcg/L |
| *diabetes* | Whether the patient has diabetes or not | Boolean (0=No, 1=Yes) |
| *ejection_fraction* | Percentage of blood leaving the heart at each contraction | Percentage |
| *high_blood_pressure* | Whether the patient has hypertension or not | Boolean (0=No, 1=Yes) |
| *platelets* | Platelets in the blood | kiloplatelets/mL	|
| *serum_creatinine* | Level of creatinine in the blood | mg/dL |
| *serum_sodium* | Level of sodium in the blood | mEq/L |
| *sex* | Female (F) or Male (M) | Binary (0=F, 1=M) |
| *smoking* | Whether the patient smokes or not | Boolean (0=No, 1=Yes) |
| *time* | Follow-up period | Days |
| *DEATH_EVENT* | Whether the patient died during the follow-up period | Boolean (0=No, 1=Yes) |


### Task

The task is to predict the survival of heart failure patients by using the above given features. The first 12 features are used the features(x) to predict the target(y). The 'DEATH_EVENT' column is used as the target variable. This is a binary classification problem where the DEATH_EVENT have values 0 and 1. 

0- indicates that the patient survived during the follow-up period
1- indicates that the patient died during the follow-up period

### Access
First , I downloaded the dataset from kaggle (https://www.kaggle.com/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv) and saved it in local .

Then I registered the dataset using the Register dataset option in Datasets tab.

The dataset is registered in Azure Machine Learning Studio:
![register-dataset](https://github.com/gaya3arul/nd00333-capstone/blob/master/starter_file/screenshots-capstone/register-dataset.png) 

## Automated ML
***AutoML settings and configuration:***

![AutoML settings & configuration](https://github.com/gaya3arul/nd00333-capstone/blob/master/starter_file/screenshots-capstone/automl-settings.png)

Below you can see an overview of the `automl` settings and configuration I used for the AutoML run:

```
automl_settings = {"n_cross_validations": 2,
                   "primary_metric": 'accuracy',
                   "enable_early_stopping": True,
                   "max_concurrent_iterations": 4,
                   "experiment_timeout_minutes": 20,
                   "verbosity": logging.INFO
                  }
```

```
automl_config = AutoMLConfig(compute_target = compute_target,
                             task = 'classification',
                             training_data = dataset,
                             label_column_name = 'DEATH_EVENT',
                             path = project_folder,
                             featurization = 'auto',
                             debug_log = 'automl_errors.log,
                             enable_onnx_compatible_models = True,
                             blocked_models=['XGBoostClassifier'],
                             **automl_settings
                             )
```


AutoML Configuration
Here is an overview of the automl settings and configuration I used for the AutoML run:

`"n_cross_validations": 2`

This parameter sets how many cross validations to perform, based on the same number of folds (number of subsets). As one cross-validation could result in overfit, in my code I chose 2 folds for cross-validation; thus the metrics are calculated with the average of the 2 validation metrics.

`"primary_metric": 'accuracy'`

I chose accuracy as the primary metric as it is the default metric used for classification tasks.

`"enable_early_stopping": True`

It defines to enable early termination if the score is not improving in the short term. In this experiment, it could also be omitted because the _experiment_timeoutminutes is already defined below.

`"max_concurrent_iterations": 4`

It represents the maximum number of iterations that would be executed in parallel.

`"experiment_timeout_minutes": 20`

This is an exit criterion and is used to define how long, in minutes, the experiment should continue to run. To help avoid experiment time out failures, I used the value of 20 minutes.

`"verbosity": logging.INFO`

The verbosity level for writing to the log file.

`compute_target = compute_target`

The Azure Machine Learning compute target to run the Automated Machine Learning experiment on.

`task = 'classification'`

This defines the experiment type which in this case is classification. Other options are regression and forecasting.

`training_data = dataset`

The training data to be used within the experiment. It should contain both training features and a label column - see next parameter.

`label_column_name = 'DEATH_EVENT'`

The name of the label column i.e. the target column based on which the prediction is done.

`path = project_folder`

The full path to the Azure Machine Learning project folder.

`featurization = 'auto'`

This parameter defines whether featurization step should be done automatically as in this case (auto) or not (off).

`debug_log = 'automl_errors.log`

The log file to write debug information to.

`enable_onnx_compatible_models = True`

I have enabled the ONNX-compatible models. Finally also saved the  Open Neural Network Exchange (ONNX) model. Referred the notebook saved here(https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/classification-bank-marketing-all-features/auto-ml-classification-bank-marketing-all-features.ipynb)

'blocked_models=['XGBoostClassifier']'

I blocked the ensemble model XGBoostClassifier model . So the Automated ML run did not use the ensemble model XGBoostClassifier

### Results

Before building the model, the data is validated and processed.

The dataset is a balanced datasetno features were missing and no high cardinality features were detected .

Data Guardrails checks in the AzureML Studio:
[data-guardrails](https://github.com/gaya3arul/nd00333-capstone/blob/master/starter_file/screenshots-capstone/data-guardrails.png)

#### Completion of the AutoML run (RunDetails widget): 

[automl-runwidget](https://github.com/gaya3arul/nd00333-capstone/blob/master/starter_file/screenshots-capstone/automl-runwidget.png)

[automl-runwidget-accuracy-gr](https://github.com/gaya3arul/nd00333-capstone/blob/master/starter_file/screenshots-capstone/automl-runwidget-accuracy-gr.png)

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
