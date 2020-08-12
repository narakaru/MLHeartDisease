import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

#Create models for spot check 
def createModelForEval():
    models.append(('RFC', RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)))
    models.append(('XGB',XGBClassifier()))
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='auto')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

#Function for evaluating spot check models
def evalModels(models, X_train, y_train, X_test, y_test ):
    global y_pred

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append(y_pred)
        names.append(name)
        print('%.2f%% %s: %f (%f)' % ((accuracy * 100.0), name, y_pred.mean(), y_pred.std()))
        
        conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        print('Confusion matrix:\n-----------------\n', conf_mat)
        plotCM(conf_mat)

#Function for plot the confusion matrix
def plotCM(conf_mat):
    labels = ['Class 0', 'Class 1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()

#Function for exploring dataset 
def exploreDataSet():
    print("\nHead\n",df_train.head())
   
    print("\nWe have {} records of data with {} variables".format(*df_train.shape))
    print("\nDataset description")
    print(round(df_train.describe()))
    #Find the binary class counts to sort any imbalance if detected.
    print("\nTarget Class Distribution")
    target_count = df_train.DEATH_EVENT.value_counts()
    print('DEATH_EVENT 0:', target_count[0])
    print('DEATH_EVENT 1:', target_count[1])
    print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

    #Look at the data types of each column
    print("\nData Types of Each Column")
    print(df_train.dtypes)
    
    #Count the empty (NaN, NAN, na) values in each column
    print("\nCheck to see column with NaN, NAN or na values for cleaning data")
    print(df_train.isna().sum())

    #Plot the target_count using a bar chart
    target_count.plot(kind='bar', title='Count (DEATH_EVENT)');
    plt.show()


#Load dataset to the dataframe   
dataF = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv')
feat_labels = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time','DEATH_EVENT']

#Prepare the train and test dataset
y = df_train.DEATH_EVENT
X = df_train.drop('DEATH_EVENT', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#Call the function to describe the dataset
exploreDataSet()

print("\nSpot Check Algorithms\nEvaluate each model in turn")
#Create models for evaluation
models = []

# Spot check algorithms using the models created
createModelForEval()

# evaluate each model in turn
results = []
names = []
print("\nModel Evaluation")
print("\nAccuracy  Name   Mean   Std")
evalModels(models, X_train, y_train, X_test, y_test )

# Class count
count_class_0, count_class_1 = df_train.DEATH_EVENT.value_counts()
print(count_class_0)
print(count_class_1)

# Divide by class
df_class_0 = df_train[df_train['DEATH_EVENT'] == 0]
df_class_1 = df_train[df_train['DEATH_EVENT'] == 1]

df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('\nRandom over-sampling:')
print(df_test_over.DEATH_EVENT.value_counts())

df_test_over.DEATH_EVENT.value_counts().plot(kind='bar', title='Count (DEATH_EVENT)');
plt.show()

#----------------------------AFTER OVER SAMPLING-------------------------------
#Prepare the train and test dataset after over sampling
y = df_test_over.DEATH_EVENT
X = df_test_over.drop('DEATH_EVENT', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("\nSpot Check Algorithms\nRe-evaluate each model in turn after over sampling")

# Re-evaluate each model in turn after oversampling
results = []
names = []
print("\nAccuracy  Name   Mean   Std")
evalModels(models, X_train, y_train, X_test, y_test )


#------------------------Training the Selected Model Before Feature Selection------------------------------
dataF = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv')
#dataF = pd.read_csv('heart_failure_clinical_records_dataset.csv')

y = dataF.DEATH_EVENT
X = dataF.drop('DEATH_EVENT', axis=1)

# Split the data into 20% test and 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the classifier RFC
rfc = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
rfc.fit(X_train, y_train)

# Train the classifier XGB
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

print("\n\nImportant Feature Rating\n-------------------------------")

# Print the name and gini importance of each feature
for feature in zip(feat_labels, rfc.feature_importances_):
    print(feature)

# Create a selector object that will use the random forest classifier to identify
# features that have an importance of greater than or equal to 0.08
sfm = SelectFromModel(rfc, threshold=0.08)

# Train the selector
sfm.fit(X_train, y_train)

print("\n\nSelected Important Feature List\n------------------------")

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])

# Transform the data to create a new dataset containing only the most important features
# Apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

print("\nShape of the train dataset after feature reduction")
print(X_important_train.shape)

print("\nShape of the test dataset after feature reduction")
print(X_important_test.shape)

def RFC(): 
    # Create a new random forest classifier for the most important features
    rfc_important = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)

    # Train the new classifier on the new dataset containing the most important features
    rfc_important.fit(X_important_train, y_train)

    # Apply The Full Featured Classifier To The Test Data
    y_pred = rfc.predict(X_test)

    # View The Accuracy Of Our Full Feature (all Features) Model
    print("\nXGBoost Model")
    print("The Accuracy Of Full Feature (all Features) Model")
    print(format(accuracy_score(y_test, y_pred),"0.2f"))

    # Apply The Important Feature Classifier To The Test Data
    y_important_pred = rfc_important.predict(X_important_test)

    # View The Accuracy Of Our Limited Feature (importance > 0.1 Features) Model
    print("\nThe Accuracy Of Limited Feature (importance > 0.08 Features) Model")
    print(format(accuracy_score(y_test, y_important_pred),"0.2f"))
    # save the model to disk
    filename = 'finalized_rfcmodel.sav'
    pickle.dump(rfc, open(filename, 'wb'))


def XGB():
    # Create a new random forest classifier for the most important features
    xgb_important = XGBClassifier()

    # Train the new classifier on the new dataset containing the most important features
    xgb_important.fit(X_important_train, y_train)

    # Apply The Full Featured Classifier To The Test Data
    y_pred = xgb.predict(X_test)

    # View The Accuracy Of Our Full Feature (all Features) Model
    print("\nXGBoost Model")
    print("The Accuracy Of Full Feature (all Features) Model")
    print(format(accuracy_score(y_test, y_pred),"0.2f"))

    # Apply The Full Featured Classifier To The Test Data
    y_important_pred = xgb_important.predict(X_important_test)

    # View The Accuracy Of Our Limited Feature (importance > 0.1 Features) Model
    print("\nThe Accuracy Of Limited Feature (importance > 0.08 Features) Model")
    print(format(accuracy_score(y_test, y_important_pred),"0.2f"))

    # save the model to disk
    filename = 'finalized_xgbmodel.sav'
    pickle.dump(xgb_important, open(filename, 'wb'))


RFC()
XGB()



