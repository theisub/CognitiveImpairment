import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from tkinter import *


class Table: 
	
	def __init__(self,root): 
		
		for i in range(total_rows):
			for j in range(total_columns): 
				self.entry = Entry(root, width=4, fg='blue',font=('Arial',16,'bold'))
				self.entry.grid(row=i, column=j)
				self.entry.insert(END, New_X[i][j]) 



def multiclass_roc_auc_score(y_test,y_pred,average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)



FullDataset = pd.read_csv('File2_filtered.csv')
names = FullDataset.iloc[::,60:82].values

X = FullDataset.iloc[::,60:82].values[:len(FullDataset)-28]
np.set_printoptions(linewidth=120)  # default 75
print(FullDataset.columns.values[60:82])
#X = StandardScaler().fit_transform(X)

y = FullDataset.iloc[::,10].values[:len(FullDataset)-28]

print(len(X))
print(len(y))

for index,row in enumerate(X):
    if (np.all(np.isfinite(row)) == False or np.any(np.isnan(row)) == True ):
        print(index)
        X=np.delete(X,(index),axis=0)
        y= np.delete(y,(index),axis=0)


print(len(X))
print(y)


pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)
print(X)


New_X = FullDataset.iloc[::,60:82].values[len(FullDataset)-28:len(FullDataset)-27]
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

classifier = RandomForestClassifier(n_estimators = 10)
classsifier = classifier.fit(X_train, y_train)


predicted = classifier.predict(X_test)
predictNew = classifier.predict(New_X)



total_rows = len(New_X) 
total_columns = len(names[0]) 

root = Tk() 
t = Table(root) 
root.mainloop() 


print(confusion_matrix(y_test,predicted))
print('Accuracy score: ',accuracy_score(y_test,predicted))
roc_auc_sum = multiclass_roc_auc_score(y_test,predicted)
print('AUC Score: ', roc_auc_sum)
print('Report: ')
rep_dict = classification_report(y_test,predicted,output_dict = True)
r2 = rep_dict['1']['f1-score']
r3 = rep_dict['2']['f1-score']
print(classification_report(y_test,predicted))


print('Overall auc', roc_auc_sum)
print('R2', r2)
print('R3', r3)



plt.scatter(X_pca[:,0],X_pca[:,1],c=y)
plt.show()

imprtnc = classifier.feature_importances_
fig,ax= plt.subplots()
width = 0.4
ind = np.arange(len(imprtnc))
ax.barh(ind,imprtnc,width,color='green')
ax.set_yticks(ind+width/10)
ax.set_yticklabels(FullDataset.columns.values[60:82],minor=False)
plt.title('Значимость характеристик')
plt.xlabel('Значимость')
plt.ylabel('Хар-ка')
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5,4.5,forward=True)
plt.show()