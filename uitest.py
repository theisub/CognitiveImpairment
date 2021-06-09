import sys
from tkinter.constants import NONE

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtWidgets import QToolBar, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QTableView, QGridLayout,QStyledItemDelegate, QFileDialog
from PyQt5.QtCore import QSize, Qt, QVariant
from PyQt5 import QtGui
from functools import partial
import mplcursors
from numpy.core.numeric import tensordot
from scipy.optimize import curve_fit

import numpy as np
import pymongo as mong
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor

data = {'col1':['1','2','3','4'],
        'col2':['1','2','1','3'],
        'col3':['1','1','2','1']}

def calculate_result(arr):
    mean = np.mean(arr)
    print(mean)
    if (mean >1.55):
        print('Предварительный диагноз: Легкое когнитивное нарушение')
    elif (mean >0.55):
        print('Предварительный диагноз: субъективное когнитивное нарушение')
    elif (mean < 0.45):
        print('Отклонений не обнаружено')
    else:
        print('Предварительный диагноз не может быть поставлен на текущих данных из-за возможной принадлежности к двум группам диагноза одновремено. Измените набор данных')


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
        print(index.column())
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
    
    def get_data(self):
        return self.data
    
    def get_classifier(self):
        return self.classifier
    

    
    def read_importfile(self,filename):

        X = pd.read_csv(filename)
        print(X)
        X = X.select_dtypes(include="number")
        X=X.dropna(axis = 1, how = 'all')
        X = X.dropna()

        print(self.classifier.important_columns)
        self.column_names = self.FullDataset.columns.values[self.FullDataset.columns!='тяжестьнаруш']


        X = X.loc[:,X.columns != 'тяжестьнаруш']
        if (self.classifier.important_columns == None):

            print(X.columns.values)
            intersection_list = [value for value in self.column_names if value in X.columns.values]
            X= X[intersection_list].values
            self.column_names = intersection_list
            
            if (self.classifier.classifier_model.n_features_ > len(intersection_list)):
                ChangedDataset = self.FullDataset[intersection_list]
                ChangedDataset = ChangedDataset.dropna(axis = 0, how = 'any')
                y = self.y[ChangedDataset.index]
                self.classifier.classifier_model = self.classifier.classifier_model.fit(ChangedDataset,y)
                print("USED SIZE OF DB - ",ChangedDataset.shape)
                print("R2-Score:",self.classifier.classifier_model.score(ChangedDataset,y))
        else:

            print(self.column_names)
            print(self.classifier.important_columns)
            self.column_names=[value for value in self.column_names if value in self.classifier.important_columns]
            print(self.column_names)
            X= X[self.column_names].values
            ChangedDataset = self.FullDataset[self.column_names]
            ChangedDataset = ChangedDataset.dropna(axis = 0, how = 'any')
            y = self.y[ChangedDataset.index]
            self.classifier.classifier_model = self.classifier.classifier_model.fit(ChangedDataset,y)
            print("USED SIZE OF DB - ",ChangedDataset.shape)
            print("R2-Score:",self.classifier.classifier_model.score(ChangedDataset,y))


        self.data  = X
        classifier = self.classifier 
        self.prediction_array = classifier.classifier_model.predict(self.data)
        self.importances = classifier.classifier_model.feature_importances_
        print(self.prediction_array)
        calculate_result(self.prediction_array)



        # choose the input and output variables
        x, y =  np.arange(0,len(self.prediction_array)),self.prediction_array
        # curve fit
        popt, _ = curve_fit(objective, x, y)
        # summarize the parameter values
        a, b, c,d = popt
        print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
        # plot input vs output
        #plt.scatter(x, y)
        # define a sequence of inputs between the smallest and largest known inputs
        self.x_line = np.arange(min(x), max(x)+2, 1)
        self.popt = popt
        # calculate the output for the range
        y_line = objective(self.x_line, a, b, c,d)
        # create a line plot for the mapping function
        #plt.plot(x_line, y_line, '--', color='red')
        #plt.show()
        
        

    def read_dbfile(self, filename):
        FullDataset = pd.read_csv(filename)
        FullDataset = FullDataset.select_dtypes(include="number")
        self.FullDataset = FullDataset
        y = FullDataset.iloc[::,FullDataset.columns == 'тяжестьнаруш'].values
        '''
        for i in range(0,len(y)):
            if (y[i]== 1):
                y[i] = 2
            elif (y[i]==2):
                y[i]=1
        '''
        
        self.y = y
        
        self.patient_info = y # юзался для фамилий изначально, теперь диагнозы

        data = FullDataset.loc[:,FullDataset.columns != 'тяжестьнаруш']
        data = data.dropna(axis = 0, how = 'any')
        self.data = data

        #FullDataset = FullDataset.dropna()

        self.column_names = data.columns

        y = y[data.index]
        X = data.values
        #self.FullDataset = FullDataset
        np.set_printoptions(linewidth=120)  # default 75
        print(self.column_names)
        #X = StandardScaler().fit_transform(X)
        

        print(len(X))
        print(len(y))


        #self.historical = X

        # выкинуть в функцию старт 
        classifier = RandomForestRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                      max_depth=9, max_features='sqrt', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0,
                      min_impurity_split=None, min_samples_leaf=5,
                      min_samples_split=7, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=-1, oob_score=False,
                      random_state=7207, verbose=0, warm_start=False)
        
        self.classifier.classifier_model = classifier.fit(X, y)
        print("USED SIZE OF DB - ",X.shape)
        print("R2-Score:",self.classifier.classifier_model.score(X,y))



        # выкинуть в функцию финиш



class Controller:
    def __init__(self,view, model):
        self._view = view
        self._model = model
        self._connectSignals()

    def _filltable(self,data):
        
        self._view.model.clear()
        testik = np.empty(0,dtype=str)

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

            self._view.model.appendRow(items)
        
        self._model.classifier.important_columns = list(set(testik)) if testik.size!=0 else None
        self._view.model.setHorizontalHeaderLabels(list(self._model.column_names))
        self._view.table.setModel(self._view.model)
        delegate = ColorDelegate(self._view.table)

        #for column in range(0,self._view.model.columnCount()):
        #    self._view.table.setItemDelegateForColumn(column,delegate,o)
        self._model.setData(self._model.index(2,2),QVariant(QtGui.QBrush(QtGui.QColor(218, 94, 242))))

        self._view.table.show()
        
        
        
    def _importDb(self,filename):
        #fname = QFileDialog.getOpenFileName(None, 'Open file', 
        # '',"Csv files (*.csv)")
        #print(fname[0])
        self._model.read_dbfile(filename)
        print(self._model.data)
        self._filltable(self._model.data.values)
        client = mong.MongoClient('localhost',27017)
        db = client['CognitiveImpairment']
        df = pd.DataFrame(data=self._model.FullDataset.values,columns=self._model.FullDataset.columns)
        series_collection = db['CognitiveImpairment']
        series_collection.drop()
        test = series_collection.insert_many(df.to_dict('records'))
        print(test)

    
    def _importFile(self,filename):
        fname = QFileDialog.getOpenFileName(None, 'Open file', 
         '',"Csv files (*.csv)")
        print(fname[0])
        self._model.read_importfile(fname[0
        ])
        print(self._model.data)
        self._filltable(self._model.data)

    def _exportModel(self,filename):
        fname = QFileDialog.getSaveFileName(None,'Save file','','Model files (*.pkl)')
        joblib.dump(self._model.classifier, fname[0], compress=9)
        

    def _plotGraph(self):
        if self._model.prediction_array is not None:
            plt.plot(self._model.prediction_array,marker='o',color='blue')
            plt.xlabel('Номер визита',fontsize=10)
            #plt.ylabel('Оценка диагноза 0 - норма, 1-скн, 2-лкн',fontsize=10)
            mplcursors.cursor(hover=True)
            a,b,c,d = self._model.popt
            y_line = objective(self._model.x_line, a, b, c,d)
            next_visits_ctr = 3
            forecast = np.arange(len(self._model.prediction_array),len(self._model.prediction_array)+next_visits_ctr)
            predictions = objective(np.asarray(forecast),a,b,c,d)
            plt.plot(forecast,predictions,'--',markerfacecolor='none',marker='o',color='blue')
        # create a line plot for the mapping function
            plt.plot(self._model.x_line, y_line, '--', color='red')
            
            plt.ylim(0,2)
            plt.show()
    def _importModel(self):
        fname = QFileDialog.getOpenFileName(None, 'Open file', 
         '',"Model files (*.pkl)")
        print(fname[0])
        self._model.classifier = joblib.load(fname[0])
    def _showHistorical(self):
        client = mong.MongoClient('localhost',27017)
        db = client['CognitiveImpairment']
        df = pd.DataFrame(data=self._model.historical,columns=self._model.column_names)
        series_collection = db['CognitiveImpairment']
        all_data = (series_collection.find({},{"_id":0}))
        to_test= np.array([])
        patients_diagnosis = np.array([])

        for document in all_data:
            lol = np.asarray(list( map(document.get, self._model.column_names)))
            lol = lol[~np.isnan(lol).any(axis=0)]
            if lol.size == 0:
                print('empty')
                continue

            if to_test.size!=0:
                to_test = np.vstack([to_test,lol])
                patients_diagnosis = np.append(patients_diagnosis,document['тяжестьнаруш'])
            else:
                to_test=lol
                patients_diagnosis = np.append(patients_diagnosis,document['тяжестьнаруш'])

        pca = PCA(n_components=2)
        tryout = to_test[1:]
        plot_points = pca.fit_transform(to_test)
        new_points = pca.fit_transform(self._model.data)
        
        plt.scatter(plot_points[:,0],plot_points[:,1],c=patients_diagnosis)
        plt.scatter(new_points[:,0],new_points[:,1],c='red')

        plt.show()
        print(to_test)
        

    def _connectSignals(self):
        self._view.importFileBtn.clicked.connect(partial(self._importFile,"To_test.csv"))
        self._view.importDbBtn.clicked.connect(partial(self._importDb,"File2_filter (1).csv"))
        self._view.plotBtn.clicked.connect(self._plotGraph)
        self._view.importModel.clicked.connect(self._importModel)
        self._view.exportModel.clicked.connect(self._exportModel)
        self._view.showHistoricalBtn.clicked.connect(self._showHistorical)



class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(480, 240))
        self.model = QtGui.QStandardItemModel(self)
        self.setWindowTitle("forecast")    
        central_widget = QWidget(self)                  
        self.setCentralWidget(central_widget)           
 
        grid_layout = QGridLayout()             
        central_widget.setLayout(grid_layout)   
 
        self.table = QTableView(self)  
        self.table.horizontalHeader().setStretchLastSection(True)


 
        self.table.hide()

        self.importFileBtn = QPushButton('Импортировать файл ')
        self.importDbBtn = QPushButton('Импорт базы')
        self.plotBtn = QPushButton('Отобразить график стадий')
        self.importModel = QPushButton('Импортировать регрессор')
        self.exportModel = QPushButton('Экспортировать регрессор')
        self.showHistoricalBtn = QPushButton('Сравнить с историческими данными')




        
        grid_layout.addWidget(self.importFileBtn, 0, 0)
        grid_layout.addWidget(self.table, 0, 1)   
        grid_layout.addWidget(self.importDbBtn,1, 1)
        grid_layout.addWidget(self.plotBtn,2,1)
        grid_layout.addWidget(self.importModel,3,1)   
        grid_layout.addWidget(self.exportModel,4,1)
        grid_layout.addWidget(self.showHistoricalBtn,5,1)
   



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    model = Model()
    win.show()
    Controller(view=win,model=model)
    sys.exit(app.exec_())