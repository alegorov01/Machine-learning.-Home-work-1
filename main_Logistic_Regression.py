from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=pd.read_excel('adult.xlsx', sheet_name='adult')

SelectedColumns = data[['age','educational-num','race','gender',
                        'capital-gain','capital-loss','hours-per-week','income']]
#SelectedColumns = data[['age','race','gender','hours-per-week','income']]

X=pd.get_dummies(SelectedColumns,columns=['race','gender'])

del X['income']

Le=LabelEncoder()
Le.fit(data['income'])

y=pd.Series(data=Le.transform(data['income']))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)#почему random_state=42 Зачем это нужно?


model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(x_train,y_train)
predictions=model.predict(x_test)
#print(predictions)

y_pred_lr=model.predict_proba(x_test)

print('train:   ', model.score(x_train, y_train))
print('test:    ', model.score(x_test, y_test))

plt.hist(y_pred_lr[y_test==0][:,1],bins=50,alpha=0.6)
plt.hist(y_pred_lr[y_test==1][:,1],bins=50,alpha=0.6)
plt.text(0.82, 1550, 'train:   0.8184')
plt.text(0.82, 1350, 'test:   0.8248')
plt.legend(['<=50','>50'])
plt.title('age, educational-num, race, gender, capital-gain, capital-loss, hours-per-week, income')
plt.xlabel('The probability of an object getting into a class <50')
plt.ylabel('number of objects')
plt.grid()
plt.show()

