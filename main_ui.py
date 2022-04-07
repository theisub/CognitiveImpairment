from cProfile import label
from os import error
import sys
from time import perf_counter
from tkinter.constants import NONE
from turtle import color
from typing import AsyncContextManager
import logging
import graphviz
import matplotlib.patches as mpatches
from collections import Counter
from PyQt5.QtWidgets import QApplication, QFormLayout, QLineEdit, QTextEdit, QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMainWindow, QErrorMessage
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtWidgets import QToolBar, QVBoxLayout, QPushButton, QListWidget
from PyQt5.QtWidgets import QTableView, QGridLayout,QStyledItemDelegate, QFileDialog
from PyQt5.QtCore import QSize, Qt, QVariant
from PyQt5 import QtGui
from functools import partial
import mplcursors
from numpy.core.numeric import tensordot
from pymongo.common import RETRY_WRITES
from scipy.optimize import curve_fit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn import svm, tree
import numpy as np
import pymongo as mong
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix,r2_score,ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier

import matplotlib
import dtreeviz
from dtreeviz.trees import *

matplotlib.rcParams['legend.handlelength'] = 1
matplotlib.rcParams['legend.handleheight'] = 1

data = {'col1':['1','2','3','4'],
        'col2':['1','2','1','3'],
        'col3':['1','1','2','1']}

def calculate_result(arr):
    mean = np.mean(arr) + np.var(arr)

    if (mean >1.50):
        return('Предварительный диагноз: Легкое когнитивное нарушение')
    elif (mean >=0.50):
        return('Предварительный диагноз: субъективное когнитивное нарушение')
    elif (mean < 0.50):
        return('Предварительный диагноз: Отклонений не обнаружено')
    else:
        return('Предварительный диагноз не может быть поставлен на текущих данных из-за возможной принадлежности к двум группам диагноза одновремено. Измените набор данных')


def objective(x, a, b, c,d):
	return a * np.sin(b - x) + c * x**2 + d
    #return a + b * x + c


class Classifier:
    def __init__(self):
        self.classifier_model = None
        self.important_columns = None



class ColorDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        logging.debug(index.column())
        option.backgroundBrush = QtGui.QColor("red")

class Model(QtGui.QStandardItemModel):
    def __init__(self):
        QtGui.QStandardItemModel.__init__(self)

        self.data = None
        self.historical = None
        self.classifier = Classifier()
        self.column_names = None
        self.y = None
        self.patient_info = None
        self.importances = None
        self.prediction_array = None
        self.popt = None
        self.x_line = None
        self.collection_name = None
    
    def get_data(self):
        return self.data
    
    def get_classifier(self):
        return self.classifier
    

    
    def read_importfile(self,filename):
        
        X = pd.read_csv(filename)
        params = ['возраст','пол','Дата','ЦВБ' ,'Наследственность','ИнфарктМиок','СахарныйДиабет','АртериальнаяГипертония','Гипотиреоз','асс1','асс2','кшопс1','кшопс2','кшопс3','кшопс4','кшопс5','кшопс6','кшопс7','кшопс8','кшопс9','кшопс10','кшопс11','кшопсобщ','часы','кубик','забор','фаб1','фаб2','фаб3','фаб4','фаб5','фаб6','фабобщ','Бентон','тмт1', 'двсловНВ','двСКП','двсловнвобщ','двсловОВ','двслОВСКП','двслововобщ','двсловобщее']
        y_results = X.iloc[::,X.columns == 'тяжестьнаруш'].values
        
        X = X[params] 
        logging.debug(X)
        X = X.select_dtypes(include="number")
        X=X.dropna(axis = 1, how = 'all')
        X = X.dropna()

        logging.debug(self.classifier.important_columns)
        self.column_names = self.FullDataset.columns.values[self.FullDataset.columns!='тяжестьнаруш']
        X = X.loc[:,X.columns != 'тяжестьнаруш']

        test_for_pres = GradientBoostingClassifier(criterion='friedman_mse',
                          init=None, learning_rate=0.05, loss='deviance', max_depth=6,
                          max_features='sqrt', max_leaf_nodes=None,
                          min_impurity_decrease=0.3, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=10,
                          min_weight_fraction_leaf=0.0, n_estimators=80,
                          n_iter_no_change=None, 
                          random_state=3036, subsample=0.9, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
        X_train, X_test,y_train,y_test = train_test_split(X,y_results,test_size=0.2)
        test_for_pres = test_for_pres.fit(X_train, y_train)
        predicted = test_for_pres.predict(X_test)
        print(test_for_pres.feature_importances_)
        print(classification_report(y_test,predicted))

        height = test_for_pres.feature_importances_
        bars = X.columns.values[X.columns!='тяжестьнаруш']
        x_pos = np.arange(len(bars))
        indices_to_show = np.argsort(-height,axis=0)
        plt.close('all')
        
        # Create bars and choose color
        plt.bar(x_pos[:10], height[indices_to_show[:10]],width=0.4, color = (1,0.1,0.1,0.6))

        # Add title and axis names
        plt.title('10 значимых характеристик базы с 50 пациентами',fontsize=16)
        plt.xlabel('Хар-ки',fontsize=16)
        plt.ylabel('Значимость',fontsize=16)
       
        # Create names on the x axis
        plt.xticks(x_pos[:10], bars[indices_to_show[:10]],fontsize=16)
        
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(40.5, 10.5)
        fig.savefig('test2png.png', dpi=100)
        # Show graph
        plt.show()
        

        if (self.classifier.important_columns == None):

            logging.debug(X.columns.values)
            intersection_list = [value for value in self.column_names if value in X.columns.values]
            X= X[intersection_list].values
            self.column_names = intersection_list
            
            if (self.classifier.classifier_model.n_features_ > len(intersection_list)):
                ChangedDataset = self.FullDataset[intersection_list]
                ChangedDataset = ChangedDataset.dropna(axis = 0, how = 'any')
                y = self.y[ChangedDataset.index]
                self.classifier.classifier_model = self.classifier.classifier_model.fit(ChangedDataset,y)
                logging.debug("USED SIZE OF DB - ",ChangedDataset.shape)
                logging.debug("R2-Score:",self.classifier.classifier_model.score(ChangedDataset,y))
                '''
                test_for_pres = GradientBoostingClassifier(criterion='friedman_mse',
                          init=None, learning_rate=0.05, loss='deviance', max_depth=6,
                          max_features='sqrt', max_leaf_nodes=None,
                          min_impurity_decrease=0.3, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=10,
                          min_weight_fraction_leaf=0.0, n_estimators=80,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=3036, subsample=0.9, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
                X_train, X_test,y_train,y_test = train_test_split(ChangedDataset,y,test_size=0.2)
                test_for_pres = test_for_pres.fit(X_train, y_train)
                predicted = test_for_pres.predict(X_test)
                print(classification_report(y_test,predicted))
                '''
        else:

            logging.debug(self.column_names)
            logging.debug(self.classifier.important_columns)
            
            self.column_names=[value for value in self.column_names if value in self.classifier.important_columns]
            #self.column_names = list(set(self.column_names).intersection(set(self.classifier.important_columns)))
            logging.debug(self.column_names)
            self.column_names = [value for value in self.column_names if value in X.columns.values]
            X= X[self.column_names].values
            ChangedDataset = self.FullDataset[self.column_names]
            ChangedDataset = ChangedDataset.dropna(axis = 0, how = 'any')
            y = self.y[ChangedDataset.index]
            self.classifier.classifier_model = self.classifier.classifier_model.fit(ChangedDataset,y)
            logging.debug("USED SIZE OF DB - ",ChangedDataset.shape)
            logging.debug("R2-Score:",self.classifier.classifier_model.score(ChangedDataset,y))
            
            


        self.data  = X
        classifier = self.classifier 
        self.prediction_array = classifier.classifier_model.predict(self.data)
        self.importances = classifier.classifier_model.feature_importances_
        logging.debug(self.prediction_array)
        logging.debug('variance = ',np.var(self.prediction_array))
        text_result = calculate_result(self.prediction_array)
        

        
        to_draw = np.round(self.prediction_array)
        plt.title('Поставленные диагнозы для новых данных, на основе 7 характеристик')

        blues= plt.scatter(range(0,len(y_results)), y_results.astype(int),c='blue')
        reds = plt.scatter(range(0,len(to_draw)),to_draw,c='red')
        plt.yticks([1,2],['СКС','ЛКС'],fontsize=16)
        plt.xticks(fontsize=16)
        

        for i in range(0,len(to_draw)):
            lines, = plt.plot([i,i],[to_draw[i],y_results[i].astype(int)],color='black',linestyle='dashed',label='lines')

        plt.legend([blues,reds,lines],['Некорректно \nпоставленные \nдиагнозы','Корректно \nпоставленные \nдиагнозы','Выделенные \nнекорректные \nпрогнозы'],bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.xlabel('Номер обследования',fontsize=14)
        plt.ylabel('Поставленный диагноз',fontsize=14)

        print(classification_report(y_results,to_draw))
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(15, 10.5)
        fig.savefig('прогнозы.png', dpi=100)
        plt.show()
        cm = confusion_matrix(y_results,to_draw)
        disp =ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['СКН','ЛКН'])
        disp.plot(cmap='Greys')
        plt.title('Матрица неточностей поставленных диагнозов')
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(10, 10.5)
        fig.savefig('матрица неточностей.png', dpi=100)
        plt.show()

        x, y =  np.arange(0,len(self.prediction_array)),self.prediction_array
        if (len(self.prediction_array)>3):
            popt, _ = curve_fit(objective, x, y)
            a, b, c,d = popt
            #plt.scatter(x, y)
            self.popt = popt
        self.x_line = np.arange(min(x), max(x)+2, 1)

        #y_line = objective(self.x_line, a, b, c,d)

        return text_result

    def anomaly_detection(self, X, dimension_reduction_enabled = None):
        classes_nums = [0,1,2]

        svm_res = svm.OneClassSVM(nu=0.1,gamma=0.1)
        X = X.select_dtypes(include="number")
        X=X.dropna(axis = 1, how = 'all')
        X = X.dropna()
        X.index = pd.RangeIndex(len(X.index))
        X.index = range(len(X.index))

        New_Dataframe = pd.DataFrame()
        for group in classes_nums:
            Testik = X.loc[X['тяжестьнаруш']==group]
            indices = Testik.index.values
            Testik = Testik.values
            if len(Testik) == 0:
                continue
            if dimension_reduction_enabled == True:
                Testik = PCA(n_components=2).fit_transform(Testik)
            logging.debug('Кол-во у ' + str(group))
            pred = svm_res.fit_predict(Testik)
            lul = indices[np.where(pred == 1)]
            New_Dataframe = New_Dataframe.append(X.iloc[lul])
            rs = Counter(pred)
            logging.debug(rs)
        New_Dataframe.index = pd.RangeIndex(len(New_Dataframe.index))
        New_Dataframe.index = range(len(New_Dataframe.index))
        y = New_Dataframe.iloc[::,New_Dataframe.columns == 'тяжестьнаруш'].values

        return New_Dataframe
        
        

    def read_dbfile(self, FullDataset):
        FullDataset = FullDataset.select_dtypes(include="number")

        self.FullDataset = FullDataset
        y = FullDataset.iloc[::,FullDataset.columns == 'тяжестьнаруш'].values
        
        self.y = y
        
        self.patient_info = y 
        #params = ['возраст','пол','Дата','ЦВБ' ,'Наследственность','ИнфарктМиок','СахарныйДиабет','АртериальнаяГипертония','Гипотиреоз','асс1','асс2','кшопс1','кшопс2','кшопс3','кшопс4','кшопс5','кшопс6','кшопс7','кшопс8','кшопс9','кшопс10','кшопс11','кшопсобщ','часы','кубик','забор','фаб1','фаб2','фаб3','фаб4','фаб5','фаб6','фабобщ','Бентон','тмт1', 'двсловНВ','двСКП','двсловнвобщ','двсловОВ','двслОВСКП','двслововобщ','двсловобщее']
        data = FullDataset.loc[:,FullDataset.columns != 'тяжестьнаруш']
        #data = FullDataset[params]
        data = data.dropna(axis = 0, how = 'any')
        self.data = data


        self.column_names = data.columns

        y = y[data.index]
        X = data.values
        np.set_printoptions(linewidth=120)  
        logging.debug(self.column_names)
        

        if X.size == 0 or y.size == 0:
            logging.error('Файл некорректен, недостаточно данных')
            return -1 


    
        classifier = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                        init=None, learning_rate=0.05, loss='ls', max_depth=6,
                        max_features='sqrt', max_leaf_nodes=None,
                        min_impurity_decrease=0.3, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=10,
                        min_weight_fraction_leaf=0.0, n_estimators=80,
                        n_iter_no_change=None,
                        random_state=3036, subsample=0.9, tol=0.0001,
                        validation_fraction=0.1, verbose=2, warm_start=False)
        self.classifier.classifier_model = classifier.fit(X, y)
        print(self.classifier.classifier_model.feature_importances_)
        print(self.column_names.values)
        dic = zip(self.column_names.values,self.classifier.classifier_model.feature_importances_)

        height = self.classifier.classifier_model.feature_importances_
        bars = self.column_names.values
        x_pos = np.arange(len(bars))
        indices_to_show = np.argsort(-height,axis=0)
        plt.close('all')
        '''
        # Create bars and choose color
        plt.bar(x_pos[:10], height[indices_to_show[:10]],width=0.4, color = (1,0.1,0.1,0.6))

        # Add title and axis names
        plt.title('10 значимых характеристик базы с 520 пациентами',fontsize=16)
        plt.xlabel('Хар-ки',fontsize=16)
        plt.ylabel('Значимость',fontsize=16)
       
        # Create names on the x axis
        plt.xticks(x_pos[:10], bars[indices_to_show[:10]],fontsize=16)
        
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(40.5, 10.5)
        fig.savefig('test2png.png', dpi=100)
        # Show graph
        plt.show()
        
        viz = dtreeviz(self.classifier.classifier_model.estimators_[0,0],
            x_data=X,
            y_data=y,
            target_name='тяжестьнаруш',
            feature_names=data.columns,
            title="Decision Tree - Boston housing",
            show_node_labels = True)
        viz.save("decision_tree.svg")
        '''
        '''
        automl = AutoML(  algorithms=["CatBoost", "Xgboost", "LightGBM"],
            model_time_limit=2*60,
            start_random_models=10,
            hill_climbing_steps=3,
            top_models_to_improve=3,
            golden_features=True,
            features_selection=False,
            stack_models=True,
            train_ensemble=True,
            explain_level=2,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 4,
                "shuffle": False,
                "stratify": True,
            })
        automl.fit(X, y.ravel())

        #predictions = automl.predict(X_test)
        plt.show()
        plt.figure()
        tree.plot_tree(self.classifier.classifier_model.estimators_[0,0])
        '''
        # Draw graph
        '''
        for i in range(0,79):
            dot_data = tree.export_graphviz(self.classifier.classifier_model.estimators_[i,0], out_file=None, feature_names=data.columns,class_names=['Норма','СКН','ЛКН'],  filled=True)
            graph = graphviz.Source(dot_data, format="png")
            graph.render('decision_tree_graphviz{}'.format(str(i)))
        '''

        try:
            logging.debug("USED SIZE OF DB - ",X.shape)
            logging.debug("R2-Score:",self.classifier.classifier_model.score(X,y))
        except: 
            logging.error('Ошибка при попытке обучения модели')
            return -1 




class Controller:
    def __init__(self,view, model):
        self._view = view
        self._model = model
        self._connectSignals()

    def _filltable(self,data):
        
        self._view.uiWindow.model.clear()
        testik = np.empty(0,dtype=str)

        mat = data.shape[1]
        '''
        if self._model.importances is not None:
            if len(self._model.importances) < 8:
                data = self._model.FullDataset[self._model.column_names].values[30:50]
                To_skip = True
                mat = len(self._model.importances)
        
        # for row in data[::,0:mat]:
        '''
        for row in data:
            column=0    
            items = [
                QtGui.QStandardItem(str(field))
                for field in row
            ]
            if self._model.importances is not None:
                for item in items:
                    
                
                    if self._model.importances[column] / max(self._model.importances) > 0.80:
                        item.setForeground(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
                        testik = np.append(testik, self._model.column_names[column])
                    elif self._model.importances[column] / max(self._model.importances) > 0.40:
                        item.setForeground(QtGui.QBrush(QtGui.QColor(255, 140, 0)))
                        testik = np.append(testik,self._model.column_names[column])
                    elif self._model.importances[column] / max(self._model.importances) > 0.20:
                        item.setForeground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                        testik = np.append(testik,self._model.column_names[column])
                    else:
                        item.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
                    column = column +1
            self._view.uiWindow.model.appendRow(items)
        
        self._model.classifier.important_columns = list(set(testik)) if testik.size!=0 else None
        self._view.uiWindow.model.setHorizontalHeaderLabels(list(self._model.column_names))
        self._view.uiWindow.table.setModel(self._view.uiWindow.model)
        delegate = ColorDelegate(self._view.uiWindow.table)

        #for column in range(0,self._view.model.columnCount()):
        #    self._view.table.setItemDelegateForColumn(column,delegate,o)
        #self._model.setData(self._model.index(2,2),QVariant(QtGui.QBrush(QtGui.QColor(218, 94, 242))))

        self._view.uiWindow.table.show()
        
        
        
    def _createDb(self,anomaly_enabled=None):
        fname = QFileDialog.getOpenFileName(None, 'Open file', 
         '',"Csv files (*.csv)")
        logging.debug(fname[0])
        if fname[0] == '' or self._view.uiToolTab.line.text() == '':
            logging.warning('Не выбрана база')
            return
        FullDataset = pd.read_csv(fname[0])
        if anomaly_enabled == True:
            FullDataset = self._model.anomaly_detection(FullDataset,None)
        self._model.collection_name = self._view.uiToolTab.line.text()
        state = self._model.read_dbfile(FullDataset)
        if state == -1:
            logging.warning('Файл некорректен')
            return
        if self._model.data.size==0:
            logging.warning('Выбран пустой файл')
            return
        logging.debug(self._model.data)
        self._filltable(self._model.data.values)

        
        client = mong.MongoClient('localhost',27017)
        db = client['CognitiveImpairment']
        df = pd.DataFrame(data=self._model.FullDataset.values,columns=self._model.FullDataset.columns)
        series_collection = db[self._view.uiToolTab.line.text()]
        series_collection.drop()
        

        test = series_collection.insert_many(df.to_dict('records'))
        self.collection_name = self._view.uiToolTab.line.text()
        self._view.startUIWindow()

    def _importDb(self,dbname):


        client = mong.MongoClient('localhost',27017)
        dbname = self._view.uiToolTab.db_name
        db = client['CognitiveImpairment']
        if dbname is None:
            return
        series_collection = db[dbname]
        self._model.collection_name = dbname
        FullDataset = pd.DataFrame(list(series_collection.find()))

        self._model.read_dbfile(FullDataset)

        self._filltable(self._model.data.values)

        logging.debug(FullDataset)
        self._view.startUIWindow()
        self._showHistorical()

    
    def _importFile(self,filename):
        fname = QFileDialog.getOpenFileName(None, 'Open file', 
         '',"Csv files (*.csv)")
        logging.debug(fname[0])
        if fname[0] == '':
            logging.warning('error')
            return
        result_text = self._model.read_importfile(fname[0
        ])
        logging.debug(self._model.data)
        
        self._filltable(self._model.data)
        self._view.uiWindow.resultBox.clear()
        self._view.uiWindow.resultBox.setText(result_text)

    def _exportModel(self,filename):
        fname = QFileDialog.getSaveFileName(None,'Save file','','Model files (*.pkl)')
        if fname[0] == '':
            logging.warning('Не выбрано имя для экспорта')
            return
        joblib.dump(self._model.classifier, fname[0], compress=9)
        

    def _plotGraph(self):
        if self._model.prediction_array is not None:
           
            #plt.ylabel('Оценка диагноза 0 - норма, 1-скн, 2-лкн',fontsize=10)
            mplcursors.cursor(hover=True)
            


            self._view.uiWindow.figure.clear()
            ax = self._view.uiWindow.figure.add_subplot(111)
            ax.clear()
            ax.set_title('Динамика состояния пациента',fontsize='x-large')
            str_month_list = ['Норма','СКН','ЛКН']
            ax.set_yticks(range(0,3))
            ax.set_yticklabels(str_month_list)
            #ax.plot(forecast,predictions,'--',markerfacecolor='none',marker='o',color='blue')
            ax.plot(range(1,len(self._model.prediction_array)+1),self._model.prediction_array,marker='o',color='blue',label='Состояние пациента')
            for i in range(len(self._model.prediction_array)):
                print(self._model.prediction_array[i])
            ax.set_xlabel('Номер визита',fontsize='x-large')
            ax.set_xticks(range(1,len(self._model.x_line)))
            if (len(self._model.prediction_array)>3):
                a,b,c,d = self._model.popt
                y_line = objective(self._model.x_line, a, b, c,d)
                next_visits_ctr = 0
                forecast = np.arange(len(self._model.prediction_array),len(self._model.prediction_array)+next_visits_ctr)
                predictions = objective(np.asarray(forecast),a,b,c,d)
                #ax.plot(self._model.x_line+1, y_line, '--', color='red',label='Течение состояния')
            legend = ax.legend(loc='best', shadow=True, fontsize='x-large')

            ax.set_ylim(0,2)
            self._view.uiWindow.canvas.draw()

    def _importModel(self):
        fname = QFileDialog.getOpenFileName(None, 'Open file', 
         '',"Model files (*.pkl)")
        logging.debug(fname[0])
        if fname[0] == '':
            logging.warning('Не выбран файл модели')
            return

        self._model.classifier = joblib.load(fname[0])
    def _showHistorical(self):
        client = mong.MongoClient('localhost',27017)
        db = client['CognitiveImpairment']
        df = pd.DataFrame(data=self._model.historical,columns=self._model.column_names)
        series_collection = db[self._model.collection_name]
        all_data = (series_collection.find({},{"_id":0}))
        to_test= np.array([])
        patients_diagnosis = np.array([])
        OneDot = False

        for document in all_data:
            temp_arr = np.asarray(list( map(document.get, self._model.column_names)))
            temp_arr = temp_arr[~np.isnan(temp_arr).any(axis=0)]
            if temp_arr.size == 0:
                continue

            if to_test.size!=0:
                to_test = np.vstack([to_test,temp_arr])
                patients_diagnosis = np.append(patients_diagnosis,document['тяжестьнаруш'])
            else:
                to_test=temp_arr
                patients_diagnosis = np.append(patients_diagnosis,document['тяжестьнаруш'])

        pca = PCA(n_components=2)
        plot_points = pca.fit_transform(to_test)

        if len(self._model.data)==1:
            self._model.data = np.append(self._model.data,self._model.data,axis=0)
            OneDot = True


        new_points = pca.fit_transform(self._model.data)

        self._view.uiWindow.figure.clear()
        ax = self._view.uiWindow.figure.add_subplot(111)
        ax.clear()
        ax.set_title('Визуализация исторических и импортированных данных на плоскости',fontsize=10)


        colormap = np.array(['#050EED', '#E09407','#368605'])

        scatter = ax.scatter(plot_points[:,0],plot_points[:,1],c=colormap[patients_diagnosis.astype(int)])

        lkn_indices = np.where(patients_diagnosis ==2)
        if len(lkn_indices[0])>0:
            lkn_mean0 =np.mean(plot_points[lkn_indices,0])
            lkn_mean1 =np.mean(plot_points[lkn_indices,1])
            scatter = ax.plot(lkn_mean0,lkn_mean1,c='black',markerfacecolor='#368605',marker='s',markersize=15)
            
        skn_indices = np.where(patients_diagnosis ==1)
        if len(lkn_indices[0])>0:
            skn_mean0 =np.mean(plot_points[skn_indices,0])
            skn_mean1 =np.mean(plot_points[skn_indices,1])
            scatter = ax.plot(skn_mean0,skn_mean1,c='black',marker='s',markerfacecolor='#E09407',markersize=15)

        n_indices = np.where(patients_diagnosis ==0)
        if len(n_indices[0])>0:
            n_mean0 =np.mean(plot_points[n_indices,0])
            n_mean1 =np.mean(plot_points[n_indices,1])
            scatter = ax.plot(n_mean0,n_mean1,c='black',markerfacecolor='#050EED',marker='s',markersize=15)


        scatter = ax.scatter(plot_points[:,0],plot_points[:,1],c=colormap[patients_diagnosis.astype(int)])

        new_points_ctr = 0
        if self._model.importances is not None:
            scatter = ax.scatter(new_points[:,0],new_points[:,1],c='red')
            new_points_ctr = len(new_points)
            if OneDot == True:
                new_points_ctr = 1
                OneDot = False
            


        pop_norm = mpatches.Circle((0.5, 0.5), 0.2,color='#050EED', label='Норма ({} объектов)'.format(np.count_nonzero(patients_diagnosis == 0.0)))
        pop_skn = mpatches.Circle((0.5, 0.5), 0.2,color='#E09407', label='СКН ({} объектов)'.format(np.count_nonzero(patients_diagnosis == 1.0)))
        pop_lkn = mpatches.Circle((0.5, 0.5), 0.2,color='#368605', label='ЛКН ({} объектов)'.format(np.count_nonzero(patients_diagnosis == 2.0)))
        pop_new = mpatches.Circle((0.5, 0.5), 0.2,color='red', label='Импортированные данные ({} объектов)'.format(new_points_ctr))
        rect1 = mpatches.Rectangle((0,0),1,1,facecolor='None',label = 'Центры кластеров пациентов',edgecolor ='black')


       
        ax.legend(handles=[pop_norm,pop_skn,pop_lkn,pop_new,rect1],loc='best')

   

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])


        self._view.uiWindow.canvas.draw()

        #plt.show()
        

    def _connectSignals(self):
        self._view.uiWindow.importFileBtn.clicked.connect(partial(self._importFile,"To_test.csv"))
        self._view.uiWindow.importDbBtn.clicked.connect(partial(self._importDb,"File2_filter (1).csv"))
        self._view.uiWindow.plotBtn.clicked.connect(self._plotGraph)
        self._view.uiWindow.importModel.clicked.connect(self._importModel)
        self._view.uiWindow.exportModel.clicked.connect(self._exportModel)
        self._view.uiWindow.showHistoricalBtn.clicked.connect(self._showHistorical)
        self._view.uiToolTab.CPSBTN.clicked.connect(partial(self._importDb,self._view.uiToolTab.line.text()))
        self._view.uiToolTab.newDbBtn.clicked.connect(partial(self._createDb,None))





class UIWindow(QWidget):
    def __init__(self, parent=None):
        super(UIWindow, self).__init__(parent)
        # mainwindow.setWindowIcon(QtGui.QIcon('PhotoIcon.png'))
        
        # a figure instance to plot on
        self.figure = plt.figure()
   
        # this is the Canvas Widget that 
        # displays the 'figure'it takes the
        # 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        self.ToolsBTN = QPushButton('text', self)
        self.ToolsBTN.move(50, 350)
        

        self.table = QTableView(self)  
        self.table.horizontalHeader().setStretchLastSection(True)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.table.horizontalHeader().setFont(font)

 
        #self.table.hide()

        self.resultLbl = QLabel('')
        self.tableLbl = QLabel('Таблица')


        self.importFileBtn = QPushButton('Импортировать файл ')
        self.importDbBtn = QPushButton('Импорт базы')
        self.plotBtn = QPushButton('Отобразить график стадий')
        self.importModel = QPushButton('Импортировать регрессор')
        self.exportModel = QPushButton('Экспортировать регрессор')
        self.showHistoricalBtn = QPushButton('Сравнить с историческими данными')
        self.resultBox = QTextEdit('')

class UIToolTab(QWidget):
    def __init__(self, parent=None):
        super(UIToolTab, self).__init__(parent)

        self.listwidget = QListWidget()
        self.db_name = None


        client = mong.MongoClient('localhost',27017)
        db = client['CognitiveImpairment']
        collections_list  = db.collection_names()

        self.line = QLineEdit(self)
        self.CPSBTN = QPushButton("Выбрать существующую историческую базу", self)
        self.newDbBtn = QPushButton("Создать новую историческую базу", self)

        self.listwidget = QListWidget()
        self.listwidget.insertItems(len(collections_list),collections_list)
        self.listwidget.clicked.connect(self.clicked)


    def clicked(self, qmodelindex):
        self.db_name = self.listwidget.currentItem().text()
        logging.debug(self.db_name)

        #self.CPSBTN.move(100, 350)


class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        
        QMainWindow.__init__(self)

        logging.basicConfig(filename='example.log', level=logging.DEBUG)

        self.setMinimumSize(QSize(720, 480))
        self.model = QtGui.QStandardItemModel(self)
        self.setWindowTitle("forecast")
        self.uiWindow = UIWindow()
        self.uiToolTab = UIToolTab()

        self.startUIToolTab(self.model)

    def startUIWindow(self):
        
        self.Window = UIWindow(self)
        self.setWindowTitle("Подсистема анализа данных")
        self.setCentralWidget(self.Window)
 
        logging.debug(self.uiToolTab.line.text())

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)           
             
        grid_layout = QGridLayout()
        central_widget.setLayout(grid_layout)   

        grid_layout.addWidget(self.uiWindow.canvas,1,0)

        grid_layout.addWidget(self.uiWindow.resultBox, 2, 0,5,1)
        grid_layout.addWidget(self.uiWindow.resultLbl,1,0,alignment=Qt.AlignBottom)
        grid_layout.addWidget(self.uiWindow.tableLbl, 0, 1,alignment=Qt.AlignBottom)   
        grid_layout.addWidget(self.uiWindow.table, 1, 1)   
        #grid_layout.addWidget(self.uiWindow.importDbBtn,2, 1)
        grid_layout.addWidget(self.uiWindow.plotBtn,2,1)
        grid_layout.addWidget(self.uiWindow.showHistoricalBtn,3,1)
        grid_layout.addWidget(self.uiWindow.importModel,4,1)   
        grid_layout.addWidget(self.uiWindow.exportModel,5,1)
        grid_layout.addWidget(self.uiWindow.importFileBtn, 6, 1)

        self.show()

    def add_db(self):

      client = mong.MongoClient('localhost',27017)
      db = client['CognitiveImpairment']
      collections_list  = db.collection_names()

      test_name = self.uiToolTab.line.text()

      if test_name in collections_list:
          logging.debug("База существует.")
      else:
          logging.debug("You can create collection")
          created_collection  = db[test_name]
          created_collection.insert_many(data.to_dict('records'))
          
    def read_db(self,filename):

       # fname = QFileDialog.getOpenFileName(None, 'Open file', 
       #  '',"Csv files (*.csv)")
        #print(fname[0])

        self.uiWindow.model.read_dbfile(filename)
        #self._model.read_dbfile(filename)
        logging.debug(self._model.data)
        self._filltable(self._model.data.values)
        client = mong.MongoClient('localhost',27017)
        db = client['CognitiveImpairment']
        df = pd.DataFrame(data=self._model.FullDataset.values,columns=self._model.FullDataset.columns)
        series_collection = db[filename]
        series_collection.drop()
        test = series_collection.insert_many(df.to_dict('records'))
        


    def startUIToolTab(self,model):
        self.uiToolTab = UIToolTab(self)
        self.setWindowTitle("Выбор базы данных")
        self.setCentralWidget(self.uiToolTab)

        layout = QGridLayout(self)
        self.uiToolTab.setLayout(layout)
       
        layout.addWidget(self.uiToolTab.line)
        layout.addWidget(self.uiToolTab.listwidget)
        layout.addWidget(self.uiToolTab.CPSBTN)
        layout.addWidget(self.uiToolTab.newDbBtn)
        self.show()

    

   



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.uiWindow.model = Model()
    win.show()
    Controller(view=win,model=win.uiWindow.model)
    sys.exit(app.exec_())