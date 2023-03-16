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


## Summary
The best model performances (as measured by accuracy, precision, and recall) were observed in the ensemble modeling techniques. the Easy Ensemble Classifier was the clear winner. 

* Random Oversampling: 65.47% accuracy, 1% precision, 72% recall, 2% F1 score
* SMOTE Resampling: 66.2% accuracy, 1% precision, 63% recall, 2% F1 score
* ClusterCentroids: 54.4% accuracy, 1% precision, 69% recall, 1% F1 score
* SMOTEEN: 66.7% accuracy, 1% precision, 72% recall, 2% F1 score
* Balanced Random Forest Classifier: 78.8% accuracy, 4% precision, 67% recall, 7% F1 score
* Easy Ensemble Classifier: 91.9% accuracy, 7% precision, 90% recall, 14% F1 score

Based on the results, I would recommend using the Easy Ensemble Classifier to predict credit lending risk.
