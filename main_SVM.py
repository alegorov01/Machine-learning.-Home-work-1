import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=pd.read_excel('adult.xlsx', sheet_name='adult')

SelectedColumns = data[['age','educational-num','race','gender',
                        'capital-gain','capital-loss','hours-per-week','income']]

X=pd.get_dummies(SelectedColumns,columns=['race','gender'])
del X['income']

Le=LabelEncoder()
Le.fit(data['income'])

y=pd.Series(data=Le.transform(data['income']))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)#почему random_state=42 Зачем это нужно?

model=make_pipeline(StandardScaler(),SVC(gamma='auto'))
model.fit(x_train,y_train)

predictions=model.predict(x_test)

print('train:   ',model.score(x_train,y_train))
print('test:    ',model.score(x_test,y_test))


#оставляем 2 признака
SelectedColumns2 = data[['age',
                         #'educational-num',
                         #'race',
                         #'gender',
                         #'capital-gain',
                         #'capital-loss',
                         'hours-per-week',
                         'income']]
#X2=pd.get_dummies(SelectedColumns2)#,columns=['race'])
# X2=SelectedColumns2.copy()
# del X2['income']
#
# Le2=LabelEncoder()
# Le2.fit(data['income'])
# y2=pd.Series(data=Le2.transform(data['income']))
#
# x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.8, random_state=42)
# model.fit(x_train2,y_train2)
#
#
# x_min, x_max = X2['age'].min() - .5, X2['age'].max() + .5
# y_min, y_max = X2['hours-per-week'].min() - .5, X2['hours-per-week'].max() + .5
#
# h=.02
# xx, yy=np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min, y_max,h))
# z=SVC.predict(np.c_[xx.ravel(),yy.ravel()])
#
# z = z.reshape(xx.shape())
# plt.figure(1,figsize=(8,8))
# plt.pcolormesh(xx,yy,z,cmap=plt.cm.Paired)
#
# plt.scatter(X2['age'],X2['hours-per-week'],edgecolors='k',cmap=plt.cm.Paired)
# plt.xlabel('age')
# plt.ylabel('hours-per-week')
# plt.show()