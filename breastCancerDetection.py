import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))


correlation = df.corr()
print(correlation)

# splitting the data set into 
# feature data - independant data X
# target data - dependent data Y
X = df.iloc[:, 2:31].values 
Y = df.iloc[:, 1].values 

# split data into 75% training / 25% test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# data scaling - bring all features to the same level of magnitude 
# (basically turn data into 0-100 form)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# function for knn
# detect cancer is present or not 
def models(X_train,Y_train):
    #Using KNeighborsClassifier 
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    knn.fit(X_train, Y_train)

    print('K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    return knn


# to use the model function
model = models(X_train,Y_train)

# confusion matrix
# false negative - patients with cancer misdiagnosed as not having cancer
# false positive - patients with no cancer misdiagnosed as having cancer
# true positive/negative - correct diagnosis
# TP = sensitivity
# TN = specificity

# construct confusion mastrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, model.predict(X_test))

TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

print(cm)
print('Model[KNN] Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))
print()# Print a new line

# Show other ways to get the classification accuracy & other metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


print('Model - KNN')
#Check precision, recall, f1-score
print( classification_report(Y_test, model.predict(X_test)) )
#Another way to get the models accuracy on the test data
print( accuracy_score(Y_test, model.predict(X_test)))
print()#Print a new line


#Print Prediction using KNN
pred = model.predict(X_test)
print("top - prediction   /   bottom - actual classification")
print(pred)
print()
#Print the actual values
print(Y_test)

