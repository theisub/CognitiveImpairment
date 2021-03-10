import sys

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtWidgets import QToolBar, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QTableView, QGridLayout,QStyledItemDelegate, QFileDialog
from PyQt5.QtCore import QSize, Qt, QVariant
from PyQt5 import QtGui
from functools import partial

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

data = {'col1':['1','2','3','4'],
        'col2':['1','2','1','3'],
        'col3':['1','1','2','1']}


class ColorDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        print(index.column())
        option.backgroundBrush = QtGui.QColor("red")

class Model(QtGui.QStandardItemModel):
    def __init__(self):
        QtGui.QStandardItemModel.__init__(self)

        self.data = None
        self.clasiffier = None
        self.column_names = None
        self.patient_info = None
        self.importances = None
    
    def get_data(self):
        return self.data
    
    def get_classifier(self):
        return self.clasiffier
    
    def read_importfile(self,filename):
        FullDataset = pd.read_csv(filename)

        X = FullDataset.iloc[::,60:82].values
        y = FullDataset.iloc[::,10].values

        self.data  = X
        classifier = self.classsifier 
        test = classifier.predict(self.data)
        self.importances = classifier.feature_importances_
        print(test)
        

    def read_dbfile(self, filename):
        FullDataset = pd.read_csv(filename)

        self.column_names = FullDataset.columns.values[60:82]

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

        #New_X = FullDataset.iloc[::,60:82].values[len(FullDataset)-28:len(FullDataset)-27]
        #X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

        self.data = X

        classifier = RandomForestRegressor(n_estimators = 10)
        self.classsifier = classifier.fit(X, y)


class Controller:
    def __init__(self,view, model):
        self._view = view
        self._model = model
        self._connectSignals()

    def _filltable(self,data):
        
        self._view.model.clear()

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
                    elif self._model.importances[column] / max(self._model.importances) > 0.40:
                        item.setForeground(QtGui.QBrush(QtGui.QColor(255, 140, 0)))
                    elif self._model.importances[column] / max(self._model.importances) > 0.20:
                        item.setForeground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                    else:
                        item.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
                    column = column +1






            self._view.model.appendRow(items)
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
        self._filltable(self._model.data)

    def _importFile(self,filename):
        fname = QFileDialog.getOpenFileName(None, 'Open file', 
         '',"Csv files (*.csv)")
        print(fname[0])
        self._model.read_importfile(fname[0
        ])
        print(self._model.data)
        self._filltable(self._model.data)
        

    def _connectSignals(self):
        self._view.importFileBtn.clicked.connect(partial(self._importFile,"To_test.csv"))
        self._view.importDbBtn.clicked.connect(partial(self._importDb,"File2_filtered.csv"))


class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(480, 240))
        self.model = QtGui.QStandardItemModel(self)
        self.setWindowTitle("Работа с QTableWidget")    
        central_widget = QWidget(self)                  
        self.setCentralWidget(central_widget)           
 
        grid_layout = QGridLayout()             
        central_widget.setLayout(grid_layout)   
 
        self.table = QTableView(self)  
        self.table.horizontalHeader().setStretchLastSection(True)


 
        self.table.hide()

        self.importFileBtn = QPushButton('Импортировать файл ')
        self.importDbBtn = QPushButton('Импорт базы')
        
        grid_layout.addWidget(self.importFileBtn, 0, 0)
        grid_layout.addWidget(self.table, 0, 1)   
        grid_layout.addWidget( self.importDbBtn,1, 1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    model = Model()
    win.show()
    Controller(view=win,model=model)
    sys.exit(app.exec_())