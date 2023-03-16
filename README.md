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

**Oversampling with Random Oversampler**

After instantiating the RandomOverSampler command and resampling the data in pandas, we can confirm that the minority class has been enlarged to a better balance between groups.

<img width="499" alt="image" src="https://user-images.githubusercontent.com/114873837/225689284-436f0f8c-5b93-4924-b92c-ae8c792fcfec.png">


After using this resampled data to train a logistic regression model, we can then assess the performance of the model on the following measures:
* **Accuracy:** A balanced accuracy score of 65.47 indicates that the model correctly predicts 65% of the classifications
* **Precision:** The positive predicted value indicates how reliable a positive classification is. The precision of 1.00 for the low-risk class tell us that if this model classifies a candidate as low-risk, there is 100% likelihood that they belong in this category. However, the precision of 0.01 for the high-risk group is very low, meaning that if a person is categorized as high-risk for a loan, it is very unlikely that they are actually high risk.  
* **Recall/Sensitivity:** Sensitivity refers to the likelihood that actual high-risk applicants will be identified by the model. A recall of 0.72 for the high risk category indicates that the model performs well in identifying high-risk applicants. 
* **F1 Score:** An F1 score is the harmonic mean describing the relationship between precision and sensitivity. A major imbalance between the precision and recall for a class will result in a low F1 score. For this model, the F1 score for the high-risk class is low (0.02) because the model has poor precision but good recall.

The code generating the metrics above is shown here:
<img width="586" alt="image" src="https://user-images.githubusercontent.com/114873837/225749290-54d03ec5-ae13-4f75-8ce6-1e23aaf8fd79.png">

<img width="404" alt="image" src="https://user-images.githubusercontent.com/114873837/225749404-9a6fa644-3d2e-49b0-b453-b2475b8f6a39.png">

<img width="700" alt="image" src="https://user-images.githubusercontent.com/114873837/225749470-65cb22fe-d07d-49d3-a80f-cc222d076d6c.png">

**Oversampling with SMOTE**

When using the SMOTE algorithm, we achieve the same resampling count as the RandomOverSampler method. 

<img width="590" alt="image" src="https://user-images.githubusercontent.com/114873837/225695928-5437d856-2f97-45f7-9981-5146aee0412f.png">

This logistic regression model results in the following performance measures:
* **Accuracy:** A balanced accuracy score of 66.2 indicates that the model correctly predicts 66% of the classifications
* **Precision:** The precision measures are the same as the previous model. The model performs well in positively predicting the low risk classification but poorly in predicting the high risk category.
* **Recall/Sensitivity:** A recall of 0.63 for the high risk category indicates that the model does not perform as well as the previous model in correctly identifying the actual high-risk candidates.
* **F1 Score:** Like the previous model, the F1 score for the high-risk class is low (0.02) because the model has poor precision but good recall.

The code generating the metrics above is shown here:
<img width="645" alt="image" src="https://user-images.githubusercontent.com/114873837/225749633-49a8008e-bee4-4444-b326-cdf8f91be75f.png">

<img width="383" alt="image" src="https://user-images.githubusercontent.com/114873837/225749719-0a5fb23a-b687-42f1-822f-3ada8996d364.png">

<img width="670" alt="image" src="https://user-images.githubusercontent.com/114873837/225749804-c641bc05-b23a-4e2c-8063-3f433acf5f9f.png">

### Undersampling
The undersampling technique utilized the ClusterCentroids algorithm. After using the ClusterCentroids algorithm, we can confirm that the majority class has been reduced to establish a balance between the classes.

<img width="577" alt="image" src="https://user-images.githubusercontent.com/114873837/225691381-0b9947f1-b4f4-4a10-8df8-9a38f2404ae2.png">

The logistic regression model run with the undersampling produces the following performance indicators:
* **Accuracy:** A balanced accuracy score of .544 indicates that the model correctly predicts 54% of the classifications
* **Precision:** The undersampling method also demonstrates poor precision (0.01) for the high_risk group, meaning that if a candidate is flagged as high risk, it is unlikely they belong to that category. 
* **Recall/Sensitivity** A recall of 0.69 for the high risk category indicates that the model performs adequately (but not particularly well) in identifying the true high-risk borrowers.
* **F1 Score:** The F1 score of this model is the lowest yet for the high-risk group, reflecting the imbalance between precision and recall 

<img width="539" alt="image" src="https://user-images.githubusercontent.com/114873837/225749981-9639e56d-49d0-4a9f-aaa8-52d2d6cfa808.png">

<img width="393" alt="image" src="https://user-images.githubusercontent.com/114873837/225750060-38ce971f-482c-43eb-8acb-8f39fc9cc49e.png">

<img width="673" alt="image" src="https://user-images.githubusercontent.com/114873837/225750148-95482ec4-4c74-4cfa-8b94-f4a1b12de090.png">

### Deliverable 2: SMOTEENN Model

The previous models demonstrated both over- and undersamping. A SMOTEENN model combines both techniques by oversampling the minority class and then cleaning the data with undersampling. The overall goal of this strategy is to reduce outliers and separate the two classes more distinctly. The code for the SMOTEENN resampling is below. After the resampling is complete, a counter shows that the two groups are balanced but not perfectly equal.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/114873837/225761130-22cb1c83-6a58-4edc-82bb-11a6374b5647.png">

After training the logistic regression model with the resampled data and using the testing data to make predictions, the performance of the model can be assessed on the following measures:

* **Accuracy:** A balanced accuracy score of .666 indicates that the model correctly predicts 67% of the classifications
* **Precision:** Like the previous resampling methods, the SMOTEENN tecnhique also demonstrates poor precision (0.01) for the high_risk group, meaning that if a candidate is flagged as high risk, it is unlikely they belong to that category. 
* **Recall/Sensitivity** A recall of 0.72 for the high risk category indicates that the model performs well in identifying the true high-risk borrowers.
* **F1 Score:** The F1 score of this model is 0.02, reflecting there is an imbalance between precision and recall for the high-risk class 

<img width="1056" alt="image" src="https://user-images.githubusercontent.com/114873837/225769809-0290b977-11cb-4aa2-936f-d3cadcb3c4cf.png">

### Deliverable 3: Ensemble Classifiers

In an effort to strengthen the performance of models, we now test techniques for ensemble learning. Ensemble learning refers to the combination of multiple models and algorithms to improve the performance and accuracy of predictions. In this analysis, two ensemble classifiers are compared: BalancedRandomForestClassifier and EasyEnsembleClassifier.

**Ensemble with Balanced Random Forest Classifier

The performance of the model can be assessed on the following measures:

* **Accuracy:** A balanced accuracy score of .787 indicates that the model correctly predicts 79% of the classifications
* **Precision:** While the precision of this model in predicting high risk candidates is still poor (0.04), it exceeds the other resampling models. 
* **Recall/Sensitivity** A recall of 0.67 for the high risk category indicates that the model performs adequately in identifying the true high-risk borrowers.
* **F1 Score:** The F1 score of this model is 0.07, reflecting that the ensemble technique has improved the balance between precision and recall - Though it is still very low for the high-risk group.

<img width="514" alt="image" src="https://user-images.githubusercontent.com/114873837/225763684-c1511a71-daf3-4fc4-a77a-c34939b3d51d.png">

<img width="783" alt="image" src="https://user-images.githubusercontent.com/114873837/225763738-cf8b8737-911d-4876-838b-67153aa75748.png">

**Ensemble with Easy Ensemble Classifier

The performance of the model can be assessed on the following measures:

* **Accuracy:** A balanced accuracy score of .919 indicates that the model correctly predicts 92% of the classifications. This is the highest accuracy of any of the models.
* **Precision:** While the precision of this model in predicting high risk candidates is still poor (0.04), it exceeds the other resampling models. 
* **Recall/Sensitivity** A recall of 0.9 for the high risk category indicates that the model identifies the vast majority of actual high-risk loan candidates
* **F1 Score:** The F1 score of this model is 0.14 for high risk, reflecting the best ratio between precision and recall of the models.

<img width="573" alt="image" src="https://user-images.githubusercontent.com/114873837/225764596-6286a95f-5a62-45e3-a301-fb049b8d93f5.png">

<img width="802" alt="image" src="https://user-images.githubusercontent.com/114873837/225764660-43edb7df-01da-46f4-8ec8-cecd50e0d20c.png">

## Summary
The best model performances (as measured by accuracy, precision, recall, and F1 scores) were observed in the ensemble modeling techniques. the Easy Ensemble Classifier was the clear winner. 

* Random Oversampling: 65.47% accuracy, 1% precision, 72% recall, .02 F1 score
* SMOTE Resampling: 66.2% accuracy, 1% precision, 63% recall, .02 F1 score
* ClusterCentroids: 54.4% accuracy, 1% precision, 69% recall, .01 F1 score
* SMOTEEN: 66.7% accuracy, 1% precision, 72% recall, .02 F1 score
* Balanced Random Forest Classifier: 78.8% accuracy, 4% precision, 67% recall, .07 F1 score
* Easy Ensemble Classifier: 91.9% accuracy, 7% precision, 90% recall, .14 F1 score

### Recommendations

Based on the results, I would recommend using the Easy Ensemble Classifier to predict credit lending risk. Precision in predicting the high risk category was a challenge for all of these models, meaning that someone classified as "high risk" was often flagged incorrectly. It is possible that this pattern could lead to discriminatory practices against certain trustworthy loan candidates, so it is best to be mindful of improving the F1 score where possible. 
