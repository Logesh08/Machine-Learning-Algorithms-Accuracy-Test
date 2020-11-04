from os import system, name
import pandas
import matplotlib.pyplot as plt

import sklearn
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Warning Hndling
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# define our clear function 
def cls(): 
  
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear')

cls()

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=pandas.read_csv(url, names=names)

#DATASET ANALYSIS SECTION

#print(dataset.shape)
#print(dataset.head(30))
#print(dataset.describe())
#print(dataset.groupby('class').size())

#Diagram
#dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)

#Histogram
#dataset.hist()

#scatter matrix
#scatter_matrix(dataset)

#plt.show()

#ML Algorithms
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
validation_size=0.20
seed=6
X_train, X_test, Y_train, Y_test=model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring='accuracy'

models=[]
models.append(('LR ', LogisticRegression(max_iter=100000)))
models.append(('LDA ', LinearDiscriminantAnalysis()))
models.append(('KNN ', KNeighborsClassifier()))
models.append(('DTRE', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(max_iter=100000)))

results=[]
names=[]
seed=6

for name, model in models:
    kfold=model_selection.KFold(n_splits=10, random_state=seed)
    cv_results=model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg= "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)
