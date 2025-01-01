# Practical Application Assignment 17.1: Comparing Classifiers

**Contents**

* [Introduction](#Introduction)
* [How to use the files in this repository?](#how-to-use-the-files-in-this-repository)
* [Business Understanding](#Business-Understanding)
* [Data Understanding](#Data-Understanding)
* [Data Preparation](#Data-Preparation)
* [Baseline Model Comparison](#Baseline-Model-Comparison)
* [Model Comparisons](#Model-Comparisons)
* [Improving the Model](#Improving-the-Model)
* [Findings](#Findings)
* [Next steps and Recommendations](#Next-steps-and-Recommendations)
* [License](#license)

## Introduction

This repository contains the Jupyter Notebook for Application Assignment 17.1, which involves analyzing the UCI Bank Marketing Dataset. The dataset, located in the data folder under the file name [bank-data-full.csv](https://github.com/shanumansetti/PAA17_1/blob/main/data/bank-additional-full.csv), is used to build a machine learning application that predicts which customers are likely to accept a long-term deposit offer. The analysis leverages various features, including job, marital status, education, housing, and personal loan.

The primary objective of this project is to compare the performance of the following classification algorithms:

K-Nearest Neighbors (KNN)
Logistic Regression
Decision Trees
Support Vector Machines (SVM)
During the comparison, the training times and accuracy of each model will be recorded. This analysis aims to identify the most effective model for predicting customer acceptance of the long-term deposit offer, which is the focus of a phone-based marketing campaign.

## How to use the files in this repository?

The notebooks are grouped into the following categories:
* ``articles`` – More information on the data and features
* ``data`` – bank-additional-full.csv data file from Kaggle Machine Learning dataset repository used in the notebooks
* ``images`` – Image files used in Notebook and Project Description
* ``notebook`` – What Drives the Price of a Car Notebook


## Business Understanding

The business objective is to identify key features for used car prices based on the dataset provided so that Car Dealers and their Sales Team can use these key features to understand the cars that they need to have in their inventory to increase sales.

For this application, we are using classfication in Machine Learning as we are comparing classsifiers. Classification is a supervised machine learning method where the model tries to predict the correct label of a given input data. In classification, the model is fully trained using the training data, and then it is evaluated on test data before being used to perform prediction on new unseen data.

![Machine Learning Classfication Example!](./images/Machine_learning_classification_illustration_for_the_email.png)

Diagram above shows an example of a classification use case where the algorithms can learn to predict whether a given email is spam or not.

Source - https://www.datacamp.com/blog/classification-machine-learning

### Business Objective

This dataset, provided by a Portuguese banking institution, contains the results of multiple marketing campaigns. The analysis of the data indicates that the marketing efforts have been largely ineffective in encouraging customers to sign up for the long-term deposit product.

From a business perspective, the objective of this Machine Learning project is to identify key factors that could improve the success rate of these campaigns. Specifically, the project will explore questions such as:

How do loan products impact the customer success rate? For example, should we focus more on customers with housing loans?
Does having a university degree correlate with a higher success rate?
How does the contact method (e.g., cellular phone) affect the likelihood of success in promoting long-term deposit products?
By answering these questions, we aim to develop targeted strategies that could increase the effectiveness of future marketing campaigns.

## Data Understanding

Examining the data, it does not have missing values in the columns/features. Reviewing the features of the datasets like job, marital status, education, housing and personal loans to check if this has an impact on the customers where the marketing campaign was successful.

Displayed below are some charts providing visualization on some of the observations of the dataset.

![Bar Plot of Term Deposit Outcome by Education!](./images/Bar-Plot-Term-Deposit-by-Education.jpeg)


![Pie Chart of Term Deposit Outcome by Loan Type!](./images/Pie-Chart-Plot-Term-Deposit-by-Loan-Type.jpeg)

The first thing that was apparent from the provided data was that the low success rate of the marketing campaign in getting customers to sign up for the long term deposit product regardless of the features recorded for the customers (i.e., Education, Marital Status, job, contact etc.).

The one slight exception are customers with housing loan types where 52.4% signed up for the long term deposit product vs. 45.2% who did not.

An Alternative view on the data is to review number of succesful campaigns to see how features like education and job had a positive impact on the number of successful campaigns. See plots below:

<div style="display:flex">
     <div style="flex:1;padding-right:10px;">
          <img src="images/Bar-Plot-Term-Deposit-by-Education-Deposit-Yes.jpeg" width="600"/>
     </div>
     <div style="flex:1;padding-left:10px;">
          <img src="images/Bar-Plot-Term-Deposit-by-Job-Deposit-Yes.jpeg" width="600"/>
     </div>
</div>

Reviewing the plots where the customer signed up for the Bank Product/Marketing campaign was successful, you can observe the following:

- On Education, university degree folks said yes to the bank loan product
- For Job, bank had the most success with folks in admin role which is very broad, followed by Technician, then blue-collar


## Data Preparation

Apart from the imbalanced nature of the dataset, the following was done to prepare the dataset for modeling:
- Renamed "Y" feature to "deposit" to make it more meaningful
- Use features 1 - 7 (i.e., job, marital, education, default, housing, loan and contact ) to create a feature set
- Use ColumnTransformer to selectively apply data preparation transforms, it allows you to apply a specific transform or sequence of transforms to just the numerical columns, and a separate sequence of transforms to just the categorical columns
- Use LabelEncoder to encode labels of the target column
- With your data prepared, split it into a train and test set. Next, we will split the data into a training set and a test set using the train_test_split function. We will use 30% of the data as the test set


## Baseline Model Comparison

For the baseline model, decided to use a DecisionTreeClassifer which is a class capable of performing multi-class classification on a dataset. This Classifier has the ability to using different feature subsets and decision rules at different stages of classification.

This model will be compared with Logistic Regression model which is used to describe data and the relationship between one dependent variable and one or more independent variables.

Logistic Regression Machine Learning is quite fascinating and accomplishes some things far better than a Decision Tree when you possess a lot of time and knowledge. A Decision Tree's second restriction is that it is quite costly in terms of the sample size.

In training, fitting and predicting both models on the dataset, the following results were observed:

| Model Name  	        | Accuracy                              | Precision	                    | Recall 	                | F1_Score                  | Fit Time (ms) 
|-------------	        |:------------------------------------	|:-------------------------:	|:----------------------:	|:----------------------:	|:----------------------:	|
| Decision Tree       	| 0.887513                              | 0.443792                  	| 0.499954                  |  0.470202                 | 128                       |
| Logistic Regression   | 0.887594                              | 0.443797                     	| 0.500000                  |  0.470225                 | 193                       |
|             	        |                                      	|                           	|                        	|                           |                           |

Quick review of this results show that accuracy scores were very close with numbers over 85%, however the recall, precision and F1_Score were below 50%.

This means the classifier has a high number of False negatives which can be an outcome of imbalanced class or untuned model hyperparameters. More likely because of the imbalanced dataset with a higher number of Deposit = "No" records.

## Model Comparisons

In this section, we will compare the performance of the Logistic Regression model to our KNN algorithm, Decision Tree, and SVM models. Using the default settings for each of the models, fit and score each. Also, be sure to compare the fit time of each of the models.

| Model Name        	| Train Time (s)                      | Train Accuracy                | Test Accuracy 	                | 
|-------------------	|:---------------------------	|:---------------------:	|:----------------------:	|
| Logistic Regression   | 0.322                         | 0.8872047448926502        | 0.8875940762320952                 |  
| KNN                   | 55.8                          | 0.8846033783080711        | 0.8807963097839281                  |  
| Decision Tree	        | 0.376                         | 0.8911935069890049        | 0.884761673545359                 |  
| SVM                   | 24.4                          | 0.8873087995560335        | 0.8875131504410455                 |  
|                       |                               |                           |                        	| 

Looking at the results from the model comparison, Logistic Regression had the best numbers across the three metrics with lowest train time in seconds, highest training and testing accuracy scores.

## Improving the Model

This dataset is so imbalanced when you look at the Exploratory section of this Notebook. Using these features to see if we can get a higher percentage of successful sign up for long term product did not provide a positive result with the exception of customer that have housing loan with a number of 52.4%

Using Grid Search to create models with the different parameters and evaluate the performance metrics

| Model Name        	| Train Time (s)                      | Best Parameters                                          | Best Score 	                | 
|-------------------	|:---------------------------	|:-------------------------------------------------:	         |:----------------------:	|
| Logistic Regression   | 64                            | C:0.001, penalty:l2, solver: liblinear	                     | 0.8872394393842521                |  
| KNN                   | 302                           | n_neighbors: 17                                                | 0.8855397848500199                 |  
| Decision Tree         | 15.7                          | criterion: entropy, max_depth: 1, model__min_samples_leaf: 1   | 0.8872394393842521                  |  
| SVM                   | 490                           | C: 0.1, kernel: rbf                                            | 0.8872394393842521                 |  
|                       |                               |                                                                |                        	| 

For SVM, I tried a number of paramaters which took a long time (i.e., some running over 2 hours etc) and did not finish because I had to abort the processing. Finally got the following parameter to work which took over 8 minutes as shown above.

- param_grid_svc2 = { 'model__C': [ 0.1, 0.5, 1.0 ], 'model__kernel': ['rbf','linear'] }

Interesting observation in that Logistic Regression, Decision Tree and Support Vector Machines had the same best score with their different best parameters. This leaves KNN with the lowest best score. All scores were high over 85% accuracy.

## Next Steps and Recommendations

A key challenge in this analysis is the imbalanced dataset, which is heavily skewed toward unsuccessful marketing campaigns. If the model is used to identify the factors contributing to the lack of success, then the models mentioned earlier could provide valuable insights.

Alternatively, the model could help the financial institution better understand the customer profiles they should target. For example, the high success rate among customers contacted via cellular phone suggests that the bank may want to leverage modern communication channels—such as text messaging and social media platforms (e.g., Facebook, Instagram, Twitter, TikTok)—for future marketing campaigns.

![Bar Plot of Customers using Cellular Phone for Marketing Campaign!](./images/Bar-Plot-Term-Deposit-by-Contact-Deposit-Yes.jpeg)
