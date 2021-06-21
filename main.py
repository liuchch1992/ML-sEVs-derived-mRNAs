import csv
import numpy
import pandas as pd


data_file='./data.csv'

data=pd.DataFrame(pd.read_csv(data_file))

train_data=data[data['group']==0]
test_data=data[data['group']==1]

train_x= numpy.array(train_data[['PGR','ESR1','ERBB2']])
train_y= numpy.array(train_data['class'])


test_x= numpy.array(test_data[['PGR','ESR1','ERBB2']])
test_y= numpy.array(test_data['class'])

#RF
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=12,random_state=0)

#NN
#from sklearn.neural_network import MLPClassifier
#clf = MLPClassifier(solver='adam', activation = 'relu',alpha=1e-3,
                    #hidden_layer_sizes=(100,100,100),
                    #random_state=1,verbose = True)

#SVM
#from sklearn import svm
#clf = svm.SVC(kernel='rbf',probability=True)


clf.fit(train_x, train_y)

print(clf.predict(test_x))
print(clf.predict_proba(test_x))
