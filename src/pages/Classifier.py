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





class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
    @staticmethod
    def _entropy(s):

        counts = np.bincount(np.array(s, dtype=np.int64))
        percentages = counts / len(s)

        entropy = 0
        for pct in percentages:
            if pct > 0:
                entropy += pct * np.log2(pct)
        return -entropy
    
    def _information_gain(self, parent, left_child, right_child):

        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))
    
    def _best_split(self, X, y):
  
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape
        
        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            for threshold in np.unique(X_curr):

                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                if len(df_left) > 0 and len(df_right) > 0:
                    y = df[:, -1]
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]


                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_info_gain = gain
        return best_split
    
    def _build(self, X, y, depth=0):
  
        n_rows, n_cols = X.shape
        
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            best = self._best_split(X, y)
            if best['gain'] > 0:
                left = self._build(
                    X=best['df_left'][:, :-1], 
                    y=best['df_left'][:, -1], 
                    depth=depth + 1
                )
                right = self._build(
                    X=best['df_right'][:, :-1], 
                    y=best['df_right'][:, -1], 
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'], 
                    threshold=best['threshold'], 
                    data_left=left, 
                    data_right=right, 
                    gain=best['gain']
                )
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )
    
    def fit(self, X, y):

        self.root = self._build(X, y)
        
    def _predict(self, x, tree):

        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.data_left)
        
        if feature_value > tree.threshold:
            return self._predict(x=x, tree=tree.data_right)
        
    def predict(self, X):
        return [self._predict(x, self.root) for x in X]












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






