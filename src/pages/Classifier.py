import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import seaborn as sn
import streamlit as st

st.title("--IBM attrition estimator--")
cat = ['OverTime',
'MaritalStatus',
'JobRole',
'Gender',
'EducationField',
'Department',
'BusinessTravel',
'Attrition']


data = pd.read_csv('src/datset.csv')
#data["Attrition"] = data['Attrition'].astype('category').cat.codes
data = data.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1)
for i in cat:
    data[i] = (data[i].astype('category').cat.codes).apply(np.int64)
y = data['Attrition']
X = data.drop(['Attrition'], axis=1)
tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=2000, learning_rate='auto').fit_transform(X)

fig, ax = plt.subplots()
ax.scatter(tsne[:, 0], tsne[:, 1], s= 5, c=y[:], cmap='Spectral')
st.header('0- t-SNE visualization of data disturibution')
st.pyplot(fig)




scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




clf = GaussianNB()
clf.fit(X_train, y_train)
ypred = clf.predict(X_test)
st.header('1- Naive Bayes classifier')


col1, col2 = st.columns(2)
cm = confusion_matrix(y_test, ypred)
fig, ax = plt.subplots()
sn.heatmap(cm, annot=True, fmt='g')
col1.pyplot(fig)

col2.text(classification_report(y_test, ypred, digits=4))



clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
ypred = clf.predict(X_test)


st.header('2 - KNN n=3')
col1, col2 = st.columns(2)
cm = confusion_matrix(y_test, ypred)
fig, ax = plt.subplots()
sn.heatmap(cm, annot=True, fmt='g')
col1.pyplot(fig)

col2.text(classification_report(y_test, ypred, digits=4))




clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
ypred = clf.predict(X_test)

st.header('3- Decision Tree')

col1, col2 = st.columns(2)
cm = confusion_matrix(y_test, ypred)
fig, ax = plt.subplots()
sn.heatmap(cm, annot=True, fmt='g')
col1.pyplot(fig)

col2.text(classification_report(y_test, ypred, digits=4))


clf = RandomForestClassifier(max_depth=9, random_state=2019)
clf.fit(X_train, y_train)
ypred = clf.predict(X_test)

st.header('4- Random forrest maxdepth=9')

col1, col2 = st.columns(2)
cm = confusion_matrix(y_test, ypred)
fig, ax = plt.subplots()
sn.heatmap(cm, annot=True, fmt='g')
col1.pyplot(fig)

col2.text(classification_report(y_test, ypred, digits=4))


clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2019
)
clf.fit(X_train, y_train)

ypred = clf.predict(X_test)

st.header('5- XGBoost: 500 estimators-maxdepth=9')

col1, col2 = st.columns(2)
cm = confusion_matrix(y_test, ypred)
fig, ax = plt.subplots()
sn.heatmap(cm, annot=True, fmt='g')
col1.pyplot(fig)
col2.text(classification_report(y_test, ypred, digits=4))






