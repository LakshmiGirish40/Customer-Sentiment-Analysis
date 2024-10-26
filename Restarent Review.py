import numpy as np
import pandas as pd
dataset = pd.read_csv(r"C:\Users\laksh\VS_Code\Machine_Learning\Restaurent_Review-NLP_NLTK\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps = PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)
#==========================================
# Train a classifier (e.g., Logistic Regression)
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression()
classifier1.fit(X_train, y_train)
# Make predictions
y_pred_tfidf = classifier1.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
# Confusion matrix
cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
print("Confusion matrix for TFIDF", cm_tfidf)

# Bias (Training accuracy)
bias_tfidf = classifier1.score(X_train, y_train)
print("Bias for TFIDF (Training accuracy):", bias_tfidf)

# Variance (Testing accuracy)
variance_tfidf = classifier1.score(X_test, y_test)
print("Variance for TFIDF (Testing accuracy):",variance_tfidf)
#======================================
from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
bias = classifier.score(X_train,y_train)
print("Decision Tree bias(Training Accuracy)", bias)

variance = classifier.score(X_test,y_test)
print("Decision tree variance(testing accracy)", variance)
#======================
from sklearn.neighbors import KNeighborsClassifier
#instance the model
knn = KNeighborsClassifier(n_neighbors=3)
#fit the model to the training set
knn.fit(X_train, y_train)
#predict test-set results
y_pred_knn = knn.predict(X_test)
y_pred_knn


from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test,y_pred_knn)
print(cm_knn)
bias = classifier.score(X_train,y_train)
print("KNeighborsClassifier bias (Traing Accuracy)",bias)
variance = classifier.score(X_test,y_test)
print("KNeighborsClassifier bias (Testing Accuracy)",variance)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#----------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.model_selection import GridSearchCV
# Base RandomForest model
rf_base = RandomForestClassifier(random_state=0)

# Parameter grid for hyperparameter tuning
param_grid_rf = {
    'n_estimators': [10, 30, 50, 70, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3],
    'bootstrap': [True, False]
}

# Function to tune hyperparameters
def tune_clf_hyperparameters(clf, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

# Using the function to get the best parameters
best_rf_params, best_rf_score = tune_clf_hyperparameters(rf_base, param_grid_rf, X_train, y_train)
print('RF Optimal Hyperparameters: \n', best_rf_params)

# Create and train the model with the best hyperparameters
best_rf = RandomForestClassifier(**best_rf_params, random_state=0)
best_rf.fit(X_train, y_train)

y_pred_rf = best_rf.predict(X_test)
# Evaluate the optimized model on the train data
print(classification_report(y_train, best_rf.predict(X_train)))
# Evaluate the optimized model on the test data
print(classification_report(y_test, best_rf.predict(X_test)))

cm_rf = confusion_matrix(y_test,y_pred_rf)
print("Random_forestcm_rf confusion matrix",cm_rf)
bias_rf = best_rf.score(X_train,y_train)
print("Random_forest bias (Traing Accuracy)",bias_rf)
variance_rf = best_rf.score(X_test,y_test)
print("Random_forest variance (Testing Accuracy)",variance_rf)
#=================================================================

#=====================================================
from sklearn.naive_bayes import MultinomialNB
review_tfidf_model = MultinomialNB()
review_tfidf_model = review_tfidf_model.fit(X_train,y_train)
y_pred_nb= review_tfidf_model.predict(X_test)


cm_rf = confusion_matrix(y_test,y_pred_nb)
print("Rnaive_bayes confusion matrix",cm_rf)
bias_nb = review_tfidf_model.score(X_train,y_train)
print("naive_bayesbias (Traing Accuracy)",bias_nb)
variance_nb = review_tfidf_model.score(X_test,y_test)
print("naive_bayes variance (Testing Accuracy)",variance_nb)

from sklearn.metrics import classification_report
print(classification_report(y_pred_nb,y_test))
    
#==============================================================
#naive_bayes
from sklearn.naive_bayes import MultinomialNB
review_model=MultinomialNB()
review_model.fit(X_train,y_train)
y_pred_review = review_model.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report
score = accuracy_score(y_test,y_pred_review)
print("accscore_review",score )
print(classification_report(y_test, y_pred_review))

bias_nb = review_model.score(X_train,y_train)
print("naive_bayesbias (Traing Accuracy)",bias_nb)
variance_nb =review_model.score(X_test,y_test)
print("naive_bayes variance (Testing Accuracy)",variance_nb)

#======================================

#TFIDF
X_train, X_test, y_train, y_test = train_test_split(corpus,y,test_size=0.20,random_state=0)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000,ngram_range=(1,2))

X_train = tfidf.fit_transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()


from sklearn.naive_bayes import MultinomialNB
tfidf_model=MultinomialNB()
tfidf_model.fit(X_train,y_train)
y_pred_tfidf = tfidf_model.predict(X_test)

tfidf_score = accuracy_score(y_test,y_pred_tfidf)
print("accscore_tfidf",score )
print(classification_report(y_test, y_pred_tfidf))

bias_nb = tfidf_model.score(X_train,y_train)
print("naive_bayesbias (Traing Accuracy)",bias_nb)
variance_nb =tfidf_model.score(X_test,y_test)
print("naive_bayes variance (Testing Accuracy)",variance_nb)
#===============================================================
#SVM Model Building
from sklearn.svm import SVC
svm = SVC()

from sklearn.model_selection import GridSearchCV
# Parameter grid for hyperparameter tuning
param_grid_svm = {
    'C': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 20],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 0.5, 1, 5],
    'degree': [2, 3, 4],
}

# Function to tune hyperparameters
def tune_model_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_

# Using the function to get the best parameters and model
best_svm_hyperparams, best_svm = tune_model_hyperparameters(svm, param_grid_svm, X_train, y_train)
print('SVM Optimal Hyperparameters: \n', best_svm_hyperparams)
#Evaluate the optimized model on training data
print("\nTraining Data Evaluation:")
train_predictions = best_svm.predict(X_train)
print(classification_report(y_train, train_predictions))

# Evaluate the optimized model on test data
print("\nTest Data Evaluation:")
test_predictions = best_svm.predict(X_test)
print(classification_report(y_test, test_predictions))

# Calculate and print accuracy scores for train and test sets
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
print("\nTraining Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Calculate and print bias and variance
bias = best_svm.score(X_train, y_train)
variance =best_svm.score(X_test, y_test)
print("\nSVM Bias (Training Accuracy):", bias)
print("SVM Variance (Testing Accuracy):", variance)
#==============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(criterion="gini",splitter='best',random_state=0,max_depth=8)
classifier1.fit(X_train, y_train)
y_train_pred_dt = classifier.predict(X_train)
# Predicting the Test set results
y_test_pred_dt = classifier1.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_test_pred_dt)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_test_pred_dt)
print(ac)

# Calculate and print accuracy scores for train and test sets
train_accuracy = accuracy_score(y_train,y_train_pred_dt)
test_accuracy = accuracy_score(y_test, y_test_pred_dt )
print("\nTraining Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
bias = classifier1.score(X_train, y_train)
print("\nDecision Tree  Bias (Training Accuracy):", bias)
variance = classifier1.score(X_test, y_test)
print("Decision Tree  Variance (Testing Accuracy):", variance)
#================================================================
#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=1000)
model_abc = abc.fit(X_train,y_train)
y_train_pred_abc = model_abc.predict(X_train)
y_test_pred_abc = model_abc.predict(X_test)


#AdaBoost Acccuracy
cllasi_report_abc = classification_report(y_test,y_test_pred_abc)

#Confusion matrix for AdaBoost
cm_abc = confusion_matrix(y_test,y_test_pred_abc)
cm_abc
train_accuracy = accuracy_score(y_train,y_train_pred_abc )
test_accuracy = accuracy_score(y_test, y_test_pred_abc )
print("\nTraining Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
bias = model_abc.score(X_train, y_train)
print("\nAdaBoostClassifier  Bias (Training Accuracy):", bias)
variance = model_abc.score(X_test, y_test)
print("AdaBoostClassifier  Variance (Testing Accuracy):", variance)
#=======================================================================
#GradientBoostingClassifer

from sklearn.ensemble import GradientBoostingClassifier
model_gbc = GradientBoostingClassifier()
model_gbc.fit(X_train,y_train)

#prediction Gradient Boosting Classifier
y_pred_gbc=model_gbc.predict(X_test)

print(classification_report(y_pred_gbc,y_test))

cm_gbc = confusion_matrix(y_test,y_pred_gbc)
cm_gbc
#Evaluate the optimized model on test data
print("\nTest Data Evaluation:")
test_pred_GBC =model_gbc.predict(X_test)
print(classification_report(y_test, y_pred_gbc))
print("\nTrain Data Evaluation:")
train_pred_GBC =model_gbc.predict(X_train)
print(classification_report(y_test, test_pred_GBC))

# Calculate and print accuracy scores for train and test sets
train_accuracy = accuracy_score(y_train, train_pred_GBC)
test_accuracy = accuracy_score(y_test, test_pred_GBC)
print("\nTraining Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Calculate and print bias and variance
bias =model_gbc.score(X_train, y_train)
variance =model_gbc.score(X_test, y_test)
print("\nGradientBoostingClassifier Bias (Training Accuracy):", bias)
print("GradientBoostingClassifier Variance (Testing Accuracy):", variance)
#========================================================
#Xgboost
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb

#XGb model fit
model_xgb.fit(X_train,y_train)
modelxgb = model_xgb.fit(X_train,y_train)

#Predict XGBC
ytest_pred_xgb=model_xgb.predict(X_test)
print("\nTest Data Evaluation:")
ytrain_pred_xgb=model_xgb.predict(X_train)
print("\nTrain Data Evaluation:")
#XGBC Classification report
class_rept_xgb = print(classification_report(ytest_pred_xgb,y_test))

pd.to_csv('trained_model_xgb.csv', index=False)
import os
print(os.getcwd())

# Calculate and print accuracy scores for train and test sets
train_accuracy = accuracy_score(y_train, ytrain_pred_xgb)
test_accuracy = accuracy_score(y_test, ytest_pred_xgb)
print("\nTraining Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Calculate and print bias and variance
bias =model_gbc.score(X_train, y_train)
variance =model_gbc.score(X_test, y_test)
print("\nXGBClassifier Bias (Training Accuracy):", bias)
print("XGBClassifier Variance (Testing Accuracy):", variance)

#==============================================================
#Naive Bayes and Gradient Boosting are your best options based on the results.
'''Gradient Boosting Classifier (GBC):

Training Accuracy: 0.8475 (Moderate bias)
Testing Accuracy: 0.735 (Moderate variance)
Observations: GBC has a better balance between training and testing accuracy than most other models and generalizes well to unseen data.
'''
'''Random Forest (RF):

Training Accuracy: 0.8025 (Moderate bias)
Testing Accuracy: 0.715 (Moderate variance)
Observations: The Random Forest model has a decent balance between training and testing accuracy. Though not overfitting as much as KNN or the Decision Tree, its performance can still be improved.
'''