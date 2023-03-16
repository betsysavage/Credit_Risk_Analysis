# Credit_Risk_Analysis

## Overview
Machine learning models are often used by financial institutions to help predict risks when making loans. In this challenge, a hypothetical client, Fast-Lending, is a peer-to-peer lending company looking to utilize an effective machine learning model to quickly and reliably identify loan applicants who may be considered "low-risk" or "high-risk." It is believed that by making accurate predictions, the company can reduce the number of people who default on their loans.

### Purpose
The purpose of this analysis is to use Python to build and assess the performance of various machine learning models to predict credit risk. The model options are built using a variety of techniques such as resampling (oversampling and undersampling) and emsemble learning. The performance - how well the model can predict default - will be assessed based on accuracy, precision, and recall.

### Resources
* "LoanStats_2019Q1" CSV file
* Jupyter notebook
* Pandas
* Scikit-learn
* Imbalance-learn

## Results
Before instantiating any machine learning models, we prepared the data for analysis by reading a certain subset of columns from the dataset, dropping null values, reformating data types (from strings and percentages to numbers), and consolidating values for loan statuses into two risk categories: low-risk and high-risk. Then we identified the feature and target values (the independent x and dependent y variables, respectively) to be used in all models. Once this initial data processing was completed, the data was ready to be split into training and testing groups to be used in the machine learning models.

### Deliverable 1: Resampling Models
When the two possible categories, or classes, within a dataset are imbalanced, one is much larger than the other. Within this dataset, the high-risk group is much smaller than the low-risk group because the percentage of people defaulting on loans is low. To ensure that the model isn't unfairly biased towards the majority class, resampling methods can be used to oversample from the underrepresented category or understample from the overrepresented category in order to achieve a fairer balance when training the model. Both resampling strategies were utilized across three types of models.
### Oversampling
The two oversampling techniques used were RandomOverSampler and SMOTE. Within the original sample of training data, the "low risk" class of the target group contained 51366 observations, while the "high risk" class of the target group contained 246 observations. 

<img width="479" alt="image" src="https://user-images.githubusercontent.com/114873837/225688660-847509fe-108f-41fa-b680-5d6efa5f11e2.png">

After instantiating the RandomOverSampler command and resampling the data in pandas, we can confirm that the minority class has been enlarged to a better balance between groups.

<img width="499" alt="image" src="https://user-images.githubusercontent.com/114873837/225689284-436f0f8c-5b93-4924-b92c-ae8c792fcfec.png">

After using this resampled data to train a logistic regression model, we generate the following results:
* A balanced accuracy score of 65.47
* 
* 


### Undersampling
The undersampling technique utilized the ClusterCentroids algorithm.


### Deliverable 2: SMOTEENN Model


### Deliverable 3: Ensemble Classifiers


## Summary
The best model performances (as measured by accuracy, precision, and recall) were observed in the ensemble modeling techniques. the Easy Ensemble Classifier was the clear winner. 

* Random Oversampling: 65.47% accuracy, 1% precision, 72% recall, .02 F1 score
* SMOTE Resampling: 66.2% accuracy, 1% precision, 63% recall, .02 F1 score
* ClusterCentroids: 54.4% accuracy, 1% precision, 69% recall, .01 F1 score
* SMOTEEN: 66.7% accuracy, 1% precision, 72% recall, .02 F1 score
* Balanced Random Forest Classifier: 78.8% accuracy, 4% precision, 67% recall, .07 F1 score
* Easy Ensemble Classifier: 91.9% accuracy, 7% precision, 90% recall, .14 F1 score

Based on the results, I would recommend using the Easy Ensemble Classifier to predict credit lending risk.
