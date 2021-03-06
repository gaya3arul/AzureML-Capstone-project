# Operationalizing Machine Learning with AutoML & AML Pipelines

## Table of contents
   * [Overview](#Overview)
   * [Architectural Diagram](#Architectural-Diagram)
   * [Key Steps](#Key-Steps)
   * [Screen Recording](#Screen-Recording)
   * [Improvement Suggestions](#Improvement-Suggestions)


## Overview

In this project, Azure Machine Learning is used to build and operationalize a classification model that 
utilizes the UCI Bank Marketing dataset (https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) to predict if a bank client will subscribe to a term deposit or not. The focus in this project will be on the operationalization of the best model that
results from an AutoML training run. 

## Architectural Diagram
![Architecture-diagram2](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/Architecture-diagram2.png)

This is the process flow that needs to followed

![architectur_diagram](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/architectur-diagram.png) 

## Key Steps
**Step 1: Authentication**

I have skipped this step as I am not authorized to create a security principal in Udacity's lab environment.When we use our own Azure Account , this step needs to be done to work on the experiment.


**Step 2: Automated ML Experiment**

The bank marketing dataset from https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv is already registered as a dataset in the Azure Machine Learning workspace .

Registered dataset:
![registered_dataset](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/Registered_dataset.png)

I created an AutoML run with the registered Bank marketing dataset. Classification has been selected and the "Explain best model" option has been checked. The "Exit criterion" has been reduced to 1 hour and the "Concurrency" has been set to 5.
![classification-model-conf](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/classification-model-conf.png)
Exploration of the data:
![dataset-explore](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/dataset-explore.png)

AutoML experiment completed successfully and VotingEnsemble is the best model with 91.98% Accuracy. Explanation of the model is also created.
Completed experiment:
![automl_run](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/automl-run.png)

Voting Ensemble is best model created from the Automated ML experiment:
![best_model](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/best-model.png)

Some best model metrics screenshots below.
The best model has an accuracy of around 92% and AUC_Weighted arount 94%.
Best Model:
![best-model-metrics](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/best-model-metrics.png)

Here you can see the graphical representation of Precision vs Recall and ROC curve:

![best-model-precision-ROC](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/best-model-precision-ROC.png)

Graphical representation of Calibration and Lift curve:
![best-model-calibration-lift](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/best-model-calibration-lift.png)


**Step 3: Model Deployment**

The best model is deployed using Azure Container Instance (ACI). Authentication has been enabled. This has been completed using the Azure Machine Learning studio (the GUI).

Deployed the best model(VotingEnsemble) with by selecting ACI(Azure Contaner Instance) as compute type and Enable Authentication:

![deploy-enable-authentication](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/deploy-enable-authentication.png)

Model is deployed and it is in healthy state, Authentication is enabled :

![deploy_model](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/deploy-model.png)


**Step 4: Enabling Logging**

We choose the best model for deployment and enable "Authentication" while deploying the model using Azure Container Instance (ACI). The executed code in logs.py enables Application Insights. "Application Insights enabled" is disabled before executing logs.py.

Applications Insights before running logs.py:
![application-insights-false](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/application-insights-false.png)

Applications Insights after running logs.py:
![application_insights_enabled](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/application-insights-enabled.png)

This screenshot has logs that are retrieved.
Retrieving Application Insights Logs:
![logs](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/logs.png)


**Step 5: Swagger Documentation**

Swagger is a set of open-source tools built around the OpenAPI Specification that can help us design, build, document and consume REST APIs. One of the major tools of Swagger is Swagger UI, which is used to generate interactive API documentation that lets the users try out the API calls directly in the browser.

The Swagger JSON file has been downloaded from the deployed model and copied to the swagger directory. The swagger.sh script has been run to download the latest Swagger container and run it on port 9080.

![swagger-bash](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/swagger-bash.png)

![swagger-bash1](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/swagger-bash1.png)


Finally, the serve.py script has been run to start a Python server on port 8000 to serve the Swagger JSON file.Swagger running on localhost showing the HTTP API methods and responses for the model.
Swagger UI now has the name of the deployed model after running the serve.py file:
![swagger_ui](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/swagger-ui.png)

We can post and get data from the webserver using Swagger. 

The following acreenshots shows various methods used and various responses we get from the server:
![http-methods](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/http-methods.png)

![http-responses](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/http-responses.png)


**Step 6: Consuming Model Endpoint**

Once the best model is deployed, I consume its endpoint using the endpoint.py script provided where I replace the values of scoring_uri and key to match the corresponding values that appear in the Consume tab of the endpoint.

The result from the model is displayed when endpoint.py script is run:
![updating-scoring-uri-key](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/updating-scoring-uri-key.png)

When two data points are passed to the server, the model has predicted that the first datapoint(bank customer) will subscribe to the term deposit and the second person will not.

Running the endpoint.py script against the API producing JSON output from the model:
![consuming-model-endpoint](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/consuming-model-endpoint.png)


**Step 7: Creating and Publishing a Pipeline**

The aml-exercise-pipelines-with-automated-machine-learning-step.ipynb notebook has been modified and run to create, publish and consume a pipeline.

ML studio showing the pipeline endpoint as Active:
![pipeline_endpoints](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/pipeline-endpoint.png)

The bankmarketing dataset with the AutoML module:
![publishe-pipeline-overview](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/publishe-pipeline-overview.png)

The bankmarketing dataset with the AutoML module completed:
![automl-module-python-sdk-completed](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/automl-module-python-sdk-completed.png)

The "Use RunDetails Widgets" in the Jupyter Notebook showing the step runs:
![RunWidgets](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/RunWidgets.png)

AutoML run completed successfully:
![automl_module_finished](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/automl_module_finished.png)

Completed Pipeline runs can be seen in the Pipelines section of Azure Machine Learning:
![completed-pipeline-runs](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/completed-pipeline-runs.png)

ML studio showing the scheduled run:
![ml-studio-showing-scheduled-run](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/ml-studio-showing-scheduled-run.png)


Benchmarking :(optional step)

A benchmark is used to create a baseline or acceptable performance measure. Benchmarking HTTP APIs is used to find the average response time for a deployed model.


Apache Benchmark is an easy and popular tool for benchmarking HTTP services. After updating the benchmark.sh file with scoring uri and key, the script is executed using bash command.

Updating benchmark with correct scoring uri and key:
![updating-benchmark](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/updating-benchmark.png)

Ran shell script benchmark.sh. It takes around 94.86 ms per request and there are no failed requests.

![Benchmarking](https://github.com/gaya3arul/operationalize-azure-ml-proj-2/blob/main/Exercise_starter_files/screenshots/benchmark.png)
**Step 8: Documentation**
Architectural diagram , Screenshots, and Screencast have been used for documentation.

The screencast covers the following. 

- The working deployed ML model endpoint
- The deployed Pipeline
- Available AutoML Model
- Successful API requests to the endpoint with a JSON payload

## Screen Recording
The screencast containing the project results has been recorded using the Camtasia and has been uploaded to YouTube: https://youtu.be/eb4cwFwbt_s.


## Improvement Suggestions
- Dataset:The dataset is imbalanced dataset. This is a classification problem and there are two classes in the target column. (yes or no) i.e people who subscribe to term deposit and people who will not. We will need to tackle the issue outlined below by introducing techniques for handling imbalanced data. This will reduce the bias towards one class and, therefore, produce a more reliable model capable of recognising both classes. Alternatively, we can use a metric that is more suitable for working with imbalanced data, for example, AUC and F1 score, to get more reliable predictions. We can also use weighted score for imbalanced data.
     Learn more about imbalanced data: https://aka.ms/AutomatedMLImbalancedData
- Model: An even better model could be developed by making use of feature engineering or hyperparameter tuning using Hyperdrive.We didnt use Deep Learning models. Using DL models can give better results.
-  Pipeline: The pipeline could be scheduled to run on a regular basis or execute in a trigger-based manner .In future, we can try parallel run in a pipeline when large dataset is used as input. It will be very useful to learn and apply that knowledge. Due to lack of time , I could not try this option. But I will definitely try this in future.


