# HR-Analysis
Binary classifier to predict an employee's promotion, thus saving HR's time and labour.  

Objective was to build a data-driven binary classifier to predict whether an employee with given credentials deserves promotion or not. 
Trained 1 CatBoost and 1 LightGBM and 2 Random Forest models on 11 Folds (44 models in total). The final prediction was major voting of these models and the F-1 score of the ensemble was 0.5203. 
Used Synthetic Minority Oversampling technique (SMOTE) to address the issue of unbalanced training labels.

Metric : F1 Score
