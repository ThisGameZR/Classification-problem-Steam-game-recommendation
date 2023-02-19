# **Project description**

My project is a machine learning-based recommendation system for Steam games. I am using an open-source dataset from Kaggle that contains information about various Steam games. As someone who enjoys playing games, I am interested in building a recommendation system that can suggest games that I may enjoy. I am using five different machine learning classifiers - **_Logistic Regression, K-Nearest Neighbors, SVM with linear kernel, SVM with RBF kernel, and SVM with polynomial kernel_** - to build and evaluate the recommendation system.

# **Classification problem**

The classification problem in this project is to predict whether a particular game would be a good recommendation for a user or not, based on the dataset I used contained information about various games available on the Steam platform, including their **_operating system compatibility, user reviews, price, and playtime_**. I extracted the following features to use as inputs for my classification models:

- win: a boolean variable indicating whether the game is compatible with the Windows operating system
- linux: a boolean variable indicating whether the game is compatible with the Linux operating system
- mac: a boolean variable indicating whether the game is compatible with the Mac operating system
- rating: a pre-categorize label `['Very Positive' 'Mostly Positive' 'Overwhelmingly Positive' 'Mixed']`
- positive_ratio: a numeric variable indicating the ratio of positive to negative user reviews for the game
- user_reviews: a numeric variable indicating the number of user reviews for the game
- price_final: a numeric variable indicating the final price of the game after any discounts
- price_original: a numeric variable indicating the original price of the game before any discounts
- discount: a numeric variable indicating the percentage discount applied to the game
- hours: a numeric variable indicating the average playtime of the game in hours

The classification problem I tackled with these features was to predict whether a given game would be recommended or not based on these features. I used the five machine learning classifiers mentioned above to train and evaluate models on this task, with the goal of finding the best-performing classifier for this recommendation system.

# **Data Preprocessing**

In order to use the data in our machine learning models, we need to preprocess it to ensure that it is in a format that the models can work with. Specifically, we need to perform the following preprocessing steps:

- **Label Encoding:** We use label encoding to convert categorical variables (like 'win', 'linux', 'mac', and 'rating') to numerical values. This is necessary because machine learning models can only work with numerical data. We use the `LabelEncoder` class from scikit-learn to perform this step.

```from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['win'] = le.fit_transform(X['win'])
X['linux'] = le.fit_transform(X['linux'])
X['mac'] = le.fit_transform(X['mac'])
X['rating'] = le.fit_transform(X['rating'])
```

# **Model Performances**

### **KNN - K-Nearest Neighbors**

```
[[ 686 2441]
 [ 826 16047]]

Precision    Recall  F1-Score   Support
False       0.45      0.22      0.30      3127
True        0.87      0.95      0.91     16873

Accuracy                           0.84     20000
Macro Avg   0.66      0.59      0.60     20000
Weighted Avg 0.80      0.84      0.81     20000

AUC: 0.7165078073141112
```

The performance of the KNN model can be evaluated based on several metrics. In the confusion matrix, the model predicted 686 true negatives (TN), 2441 false positives (FP), 826 false negatives (FN), and 16047 true positives (TP).

The precision of the model for predicting the positive class (is recommended) is 0.87, meaning that out of all instances that the model predicted as positive, 87% of them are actually positive. The recall for the positive class is 0.95, meaning that the model was able to correctly identify 95% of all positive instances. The F1 score, which is a balance between precision and recall, is 0.91 for the positive class.

The accuracy of the model is 0.84, meaning that it correctly classified 84% of all instances in the test set. The macro average F1 score, which is the unweighted mean of the F1 scores for both classes, is 0.6. The weighted average F1 score, which takes into account class imbalance, is 0.81.

The AUC score of the model is 0.72, which indicates that the model's ability to distinguish between positive and negative instances is moderately good. Overall, the KNN model's performance seems reasonable but there is still room for improvement.

### **LR - Logistic Regression**

```
[[   90  3037]
 [  109 16764]]
              precision    recall  f1-score   support

       False       0.45      0.03      0.05      3127
        True       0.85      0.99      0.91     16873

    accuracy                           0.84     20000
   macro avg       0.65      0.51      0.48     20000
weighted avg       0.78      0.84      0.78     20000

AUC: 0.7229217572667201
```

For the logistic regression model, the confusion matrix shows that there were 90 true negatives and 3037 false positives, and 109 false negatives and 16764 true positives. This indicates that the model correctly predicted the majority of the recommended games (true positives) but had a high number of false positives, which could result in some non-recommended games being recommended.

The classification report shows that the model has a higher precision and recall for the recommended games, indicating that it is better at correctly identifying the recommended games compared to the non-recommended games. However, the overall f1-score is low for the non-recommended games, indicating that the model struggles to accurately predict these games.

The AUC score of 0.7229 indicates that the model performs better than random guessing but there is still room for improvement. Overall, it seems like the model has a bias towards recommending games, which may not be ideal for some users who prefer a more conservative recommendation approach.

### **SVM-linear**

```
[[   89  3038]
 [  106 16767]]
              precision    recall  f1-score   support

       False       0.46      0.03      0.05      3127
        True       0.85      0.99      0.91     16873

    accuracy                           0.84     20000
   macro avg       0.65      0.51      0.48     20000
weighted avg       0.79      0.84      0.78     20000

AUC: 0.7270809160653153
```

For the SVM-Linear model, we have a confusion matrix that shows the number of true positives (16767), true negatives (89), false positives (3038), and false negatives (106).

The classification report shows the precision, recall, and f1-score for both classes (True and False) as well as the support (number of samples) for each class. In this case, the model has a higher precision and recall for the True class, which means it is better at predicting that a game is recommended rather than not recommended. The f1-score is a harmonic mean of the precision and recall, which balances them and is often used to compare models.

The accuracy of the model is 0.84, which means it correctly classified 84% of the samples. The AUC score is 0.73, which means that the model has moderate discrimination power.

### **SVM-RBF**

```
[[   60  3067]
 [   45 16828]]
              precision    recall  f1-score   support

       False       0.57      0.02      0.04      3127
        True       0.85      1.00      0.92     16873

    accuracy                           0.84     20000
   macro avg       0.71      0.51      0.48     20000
weighted avg       0.80      0.84      0.78     20000

AUC: 0.5783038569651937
```

The SVM-RBF model has an accuracy of 0.84, which means that 84% of the predictions were correct. The confusion matrix shows that the model predicted 60 true negatives and 16828 true positives. However, it also predicted 3067 false negatives and 45 false positives.

The precision of the model for the negative class is 0.57, which means that when the model predicted a negative class, it was correct 57% of the time. The recall or sensitivity of the model for the negative class is 0.02, which means that only 2% of actual negative instances were correctly identified by the model.

The precision of the model for the positive class is 0.85, which means that when the model predicted a positive class, it was correct 85% of the time. The recall or sensitivity of the model for the positive class is 1.00, which means that all actual positive instances were correctly identified by the model.

The weighted average F1-score is 0.78, which is the harmonic mean of precision and recall. It is a measure of model accuracy that considers both precision and recall.

The AUC score is 0.5783, which is relatively low, which means that the model's ability to distinguish between positive and negative classes is not very good.

### **SVM-Poly**

```
 [[   77  3050]
 [   79 16794]]

               precision    recall  f1-score   support

       False       0.49      0.02      0.05      3127
        True       0.85      1.00      0.91     16873

    accuracy                           0.84     20000
   macro avg       0.67      0.51      0.48     20000
weighted avg       0.79      0.84      0.78     20000

AUC: 0.665251038955764
```

For the SVM-Poly model, we can see from the confusion matrix that out of 3127 actual "False" values, the model predicted 77 as "False" and 3050 as "True". Out of 16873 actual "True" values, the model predicted 16794 as "True" and 79 as "False".

The precision of the "False" class is 0.49, indicating that when the model predicts a game as not recommended, it is correct 49% of the time. The recall of the "False" class is 0.02, indicating that out of all actual not recommended games, the model only correctly identified 2%.

The precision of the "True" class is 0.85, indicating that when the model predicts a game as recommended, it is correct 85% of the time. The recall of the "True" class is 1.00, indicating that out of all actual recommended games, the model correctly identified all of them.

The AUC score of the SVM-Poly model is 0.665, which is higher than the AUC score of the SVM-RBF model but lower than the AUC scores of the KNN, LR, SVM-Linear models. This indicates that the SVM-Poly model is better than the SVM-RBF model at distinguishing between positive and negative samples, but not as good as the other models.

# **Comparing Model Performances**

| Model      | Accuracy | Precision | Recall | F1-Score | AUC  |
| ---------- | -------- | --------- | ------ | -------- | ---- |
| KNN        | 0.84     | 0.87      | 0.95   | 0.91     | 0.72 |
| LR         | 0.84     | 0.85      | 0.99   | 0.91     | 0.72 |
| SVM-Linear | 0.84     | 0.85      | 0.99   | 0.91     | 0.73 |
| SVM-RBF    | 0.84     | 0.85      | 1.00   | 0.92     | 0.58 |
| SVM-Poly   | 0.84     | 0.85      | 1.00   | 0.91     | 0.67 |

| Model      | TN  | FP   | FN  | TP    |
| ---------- | --- | ---- | --- | ----- |
| KNN        | 686 | 2441 | 826 | 16047 |
| LR         | 90  | 3037 | 109 | 16764 |
| SVM-Linear | 89  | 3038 | 106 | 16767 |
| SVM-RBF    | 60  | 3067 | 45  | 16828 |
| SVM-Poly   | 77  | 3050 | 79  | 16794 |

Looking at the AUC scores, we can see that SVM-RBF has the lowest AUC score, followed by SVM-Poly, KNN, LR, and SVM-Linear. This suggests that SVM-RBF and SVM-Poly may not be the best models for this dataset.

When we consider the classification report for accuracy, precision, recall, and f1-score, we can see that all models have a similar accuracy rate of around 84%. However, the precision, recall, and f1-score differ significantly between the models.

Among the models, LR and SVM-Linear have similar performance in terms of precision, recall, and f1-score. SVM-Poly has the highest precision for False class, while LR and SVM-Linear have the highest recall and f1-score for False class. SVM-RBF has the lowest precision and recall for False class.

Overall, the LR and SVM-Linear models seem to have the best performance in terms of balanced precision and recall, with LR having the higher AUC score. However, it's important to note that the choice of the best model depends on the specific needs and constraints of the project.

# **Download Models**

[K-Nearest Neighbors](./models/knn_pipeline.pkl)

[Logistic Regression](./models/lr_pipeline.pkl)

[SVM with Linear Kernel](./models/svm_linear_pipeline.pkl)

[SVM with RBF kernel](./models/svm_rbf_pipeline.pkl)

[SVM with Polynomial Kernel]('./models/svm_poly_pipeline.pkl')

# **Resources**

https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam?resource=download
