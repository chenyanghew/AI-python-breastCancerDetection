import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read file
df = pd.read_csv('data.csv')
print('(Number of Rows, Number of Columns) =', df.shape)

# Check for any NULL value 
print(df.isna().sum())

print(df.head(10))

# We will be removing all the columns with NULL value
df = df.dropna(axis=1)

print(df.shape)
print(df.dtypes)
print(df.head(10))
print(df['diagnosis'].value_counts())

# Convert string value into int value
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
df.iloc[:,1]= labelEncoder.fit_transform(df.iloc[:,1].values)
print(labelEncoder.fit_transform(df.iloc[:,1].values))

# Getting the correlation of the colums
correlation = df.corr()
print(correlation)

# Visualisation of the correlation via heat map
plt.figure(figsize=(20,20))  
sns.heatmap(df.corr(), annot=True, fmt='.0%')

# splitting the data set into 
    # feature data - independant data x
    # target data - dependent data y
x = df.iloc[:, 2:31].values 
y = df.iloc[:, 1].values 

# split data into 80% training / 20% test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# data scaling - bring all features to the same level of magnitude 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(x_train, y_train)
print('K Nearest Neighbor Training Accuracy:', knn.score(x_train, y_train))

# Comparing the predicted value and the target value of the test data.
print(pd.DataFrame(list(zip(y_test, knn.predict(x_test))), columns=['target', 'predicted']))

# Construct confusion mastrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, knn.predict(x_test))

TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

print(cm)
print('KNN Testing Accuracy = "{}!"'.format((TP + TN) / (TP + TN + FN + FP)))

print()

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#Check precision, recall, f1-score
print( classification_report(y_test, knn.predict(x_test)) )
#Another way to get the models accuracy on the test data
print( accuracy_score(y_test, knn.predict(x_test)))
print()#Print a new line

#Print Prediction using KNN
pred = knn.predict(x_test)
print("top - prediction   /   bottom - actual classification")
print(pred)
print()
#Print the actual values
print(y_test)


