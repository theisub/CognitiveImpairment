from numpy.core.fromnumeric import shape
from numpy.lib.function_base import append
from numpy.lib.twodim_base import tri
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import RepeatedKFold, StratifiedKFold 
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from operator import truediv
from scipy.stats import gaussian_kde
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns

def filter_results(reg_res):
    filtered_arr = []
    for element in reg_res:
        if element >= 1.7:
            filtered_arr.append(2)
        if element >= 1 and element < 1.7:
            filtered_arr.append(1)
        if element < 1:
            filtered_arr.append(0)

    return filtered_arr

FullDataset = pd.read_csv("edit1.csv")

FullDataset = FullDataset.select_dtypes(include="number")

FullDataset = FullDataset
y = FullDataset.iloc[::,FullDataset.columns == 'тяжестьнаруш'].values
y = y


data = FullDataset.loc[:,FullDataset.columns != 'тяжестьнаруш']


column_names = data.columns
#3
column_names = ['асс1','асс2','кшопс5','кшопсобщ','ШОЛДобщ']
#2
#column_names = ['возраст','асс1','асс2','кшопс1','кшопс2','кшопс3','кшопс4','кшопс5','кшопс6','кшопс7','кшопс8','кшопс9','кшопс10','кшопс11','кшопсобщ','ШОЛД1','ШОЛД2','ШОЛД3','ШОЛД4','ШОЛД5','ШОЛД6','ШОЛДобщ']
X = data

ChangedDataset = FullDataset[column_names]
ChangedDataset = ChangedDataset.dropna(axis = 0, how = 'any')
X= ChangedDataset[column_names].values


# the histogram of the data


patient_info = y 


kf = KFold(n_splits=400)
kf.get_n_splits(X)

y = y[ChangedDataset.index].ravel()
y_predict = []
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    
    model = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
            init=None, learning_rate=0.05, loss='ls', max_depth=6,
            max_features='sqrt', max_leaf_nodes=None,
            min_impurity_decrease=0.3, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=80,
            n_iter_no_change=None,
            random_state=9999, subsample=0.9, tol=0.0001,
            validation_fraction=0.1, verbose=0, warm_start=False)
    '''
    model.fit(X,y.ravel())
    imprtnc = model.feature_importances_
    fig,ax= plt.subplots()
    width = 0.4
    ind = np.arange(len(imprtnc))

    ax.barh(ind,imprtnc,width,color='green')
    ax.set_yticks(ind+width/10)
    ax.set_yticklabels(ChangedDataset.columns.values,minor=False)
    plt.title('Значимость признаков')
    plt.xlabel('Значимость параметра дерева решений')
    plt.ylabel('Признак')
    plt.figure(figsize=(5,5))
    fig.set_size_inches(6.5,4.5,forward=True)
    plt.show()
    '''
    model.classifier_model = model.fit(X, y.ravel())
    y_predict = np.append(y_predict, model.fit(X[train_index], y[train_index]).predict(X[test_index]))


'''
sns.kdeplot(y_predict[:318],label='ЛКН')
sns.kdeplot(y_predict[318:421],label='СКН')
sns.kdeplot(y_predict[421:],label='Норма')
plt.xlabel('ИПСКС полученный в результате регрессионого анализа')
plt.ylabel('Плотность')
plt.show()
'''

filtered_array = filter_results(y_predict)

size_of_cm = len(set(filtered_array))
size_of_cm =3
np.set_printoptions(linewidth=120)  
confusion_matrix = np.zeros((size_of_cm,size_of_cm))
for i in range(0,len(filtered_array)):
    confusion_matrix[y[i],filtered_array[i]] +=1
confusion_matrix = confusion_matrix.transpose()
print(confusion_matrix)

print('Now sklearn')
print(precision_recall_fscore_support(y, filtered_array))

y_res = {"R2-Score":[],"R2-Std":[],"MAE":[],"MAE std":[]}

tp = np.diag(confusion_matrix)
prec = list(map(truediv, tp, np.sum(confusion_matrix, axis=0)))
rec = list(map(truediv, tp, np.sum(confusion_matrix, axis=1)))
sns.heatmap(confusion_matrix,annot=True,cmap='Greys', fmt='g',yticklabels=['Норма','СКC','ЛКC'],xticklabels=['Норма','СКС','ЛКС'])
plt.xlabel("Истина") 
plt.ylabel("Предварительный диагноз")
plt.show()
print ('Precision: {}\nRecall: {}\n F-Score'.format(prec, rec))
'''
for i in range(10,1000,10):
    classifier = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                    init=None, learning_rate=0.05, loss='ls', max_depth=6,
                    max_features='sqrt', max_leaf_nodes=None,
                    min_impurity_decrease=0.3, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=10,
                    min_weight_fraction_leaf=0.0, n_estimators=i,
                    n_iter_no_change=None,
                    random_state=3036, subsample=0.9, tol=0.0001,
                    validation_fraction=0.1, verbose=2, warm_start=False)
    classifier.classifier_model = classifier.fit(X, y)
    y_res_r2.append(classifier.classifier_model.score(X,y.ravel()))
    

x_step = np.arange(10,1000,10)
plt.plot(x_step,y_res_r2)
plt.show()
'''
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = GradientBoostingClassifier(criterion='friedman_mse',
                init=None, learning_rate=0.05,  max_depth=6,
                max_features='sqrt', max_leaf_nodes=None,
                min_impurity_decrease=0.3, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=10,
                min_weight_fraction_leaf=0.0, n_estimators=80,
                n_iter_no_change=None,
                random_state=3036, subsample=0.9, tol=0.0001,
                validation_fraction=0.1, verbose=0, warm_start=False)
y_predict = model.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))
unique, counts = np.unique(y_test, return_counts=True)
print('Кол-во представителей группы (Норма,СКН,ЛКН)',dict(zip(unique, counts)))
tess = confusion_matrix(y_test, y_predict)
print(np.transpose(confusion_matrix(y_test, y_predict)))

'''
'''
for i in range(0,500,20):
    model = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                    init=None, learning_rate=0.05, loss='ls', max_depth=6,
                    max_features='sqrt', max_leaf_nodes=None,
                    min_impurity_decrease=0.3, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=10,
                    min_weight_fraction_leaf=0.0, n_estimators=80,
                    n_iter_no_change=None,
                    random_state=9999, subsample=0.9, tol=0.0001,
                    validation_fraction=0.1, verbose=0, warm_start=False)

    #classifier.classifier_model = classifier.fit(X, y.ravel())
    #y_res_r2.append(classifier.classifier_model.score(X,y.ravel()))

    splits = 10
    cv = StratifiedKFold(n_splits=splits,  random_state=i,shuffle=True)
    n_scores = cross_validate(model, X, y.ravel(), scoring=['r2','neg_mean_squared_error'], cv=cv, n_jobs=-1, error_score='raise')

    print('Res for shape %',X.shape)
    print('splits - {0}, random state - {1}'.format(splits,i))
    print('sklearn R2: %.3f (%.3f)' % (np.mean(n_scores['test_r2']), np.std(n_scores['test_r2'])))
    print('sklearn nMAE: %.3f (%.3f)' % (np.mean(n_scores['test_neg_mean_squared_error']), np.std(n_scores['test_neg_mean_squared_error'])))
    y_res["R2-Score"].append(np.mean(n_scores['test_r2']))
    y_res["R2-Std"].append(np.std(n_scores['test_r2']))
    y_res["MAE"].append(np.mean(n_scores['test_neg_mean_squared_error']))
    y_res["MAE std"].append(np.std(n_scores['test_neg_mean_squared_error']))

    # evaluate the model
    model = XGBRegressor(objective='reg:squarederror')
    cv = StratifiedKFold(n_splits=splits,  random_state=i,shuffle=True)
    


    n_scores = cross_validate(model, X, y.ravel(), scoring=['r2','neg_mean_squared_error'], cv=cv, n_jobs=-1, error_score='raise')

    print('XGB R2: %.3f (%.3f)' % (np.mean(n_scores['test_r2']), np.std(n_scores['test_r2'])))
    print('XGB nMAE: %.3f (%.3f)' % (np.mean(n_scores['test_neg_mean_squared_error']), np.std(n_scores['test_neg_mean_squared_error'])))
    # fit the model on the whole dataset

    # make a single prediction
    #print("USED SIZE OF DB - ",X.shape)
    #print("R2-Score:",classifier.classifier_model.score(X,y.ravel()))

    model = LGBMRegressor()
    cv = StratifiedKFold(n_splits=splits,  random_state=i,shuffle=True)
    n_scores = cross_validate(model, X, y.ravel(), scoring=['r2','neg_mean_squared_error'], cv=cv, n_jobs=-1, error_score='raise')

    print('LGBM R2: %.3f (%.3f)' % (np.mean(n_scores['test_r2']), np.std(n_scores['test_r2'])))
    print('LGBM nMAE: %.3f (%.3f)' % (np.mean(n_scores['test_neg_mean_squared_error']), np.std(n_scores['test_neg_mean_squared_error'])))
    print('')
    # fit the model on the whole dataset
print('overall r2 = ',np.mean(y_res["R2-Score"]), ' std  = ',np.mean(y_res["R2-Std"]))
print('overall MAE = ',np.mean(y_res["MAE"]), ' std  = ',np.mean(y_res["MAE std"]))
'''