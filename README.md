# Credit_Risk_Analysis

## Overview

### Purpose
The purpose of this analysis is to assess the performance of various machine learning models to predict credit risk. 

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
