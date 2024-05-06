# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.
5. 
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Bolisetti Sanjitha
RegisterNumber: 212222040027
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df=pd.read_csv('/content/spam.csv',encoding='ISO-8859-1')
df.head()

vectorizer = CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model=svm.SVC(kernel='linear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
```

## Output:
# Head
![image](https://github.com/SanjithaBolisetti/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393633/30419de9-fb21-4da7-be4d-edaa08ac21cf)

# Kernel Model:
![image](https://github.com/SanjithaBolisetti/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393633/8518412e-5791-4abc-8777-8b2e46b96018)

# Accuracy and Classification report:
![image](https://github.com/SanjithaBolisetti/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393633/f1243254-dbff-4b36-af94-fd4d8197e83f)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
