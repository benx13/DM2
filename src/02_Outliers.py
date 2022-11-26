import streamlit as st  
import plotly.graph_objs as go
from DataFrame import DF
import numpy as np

st.set_page_config(layout="wide")

dataframe = DF()
data = DF().data
df = DF().df

def boxplot(var_select): 
    trace = go.Box(
        boxpoints='outliers',
        y=list(dataframe.data.get(var_select)),
        name=var_select, opacity = 0.8, marker=dict(
        color='gold', line=dict(color='#000000',width=1)))

    
    layout = dict(xaxis=dict(), yaxis=dict(title= 'Count'))
    fig = go.Figure(data=[trace], layout=layout)
    
    return fig

def plot_col(col):
    st.subheader("Outliers")

    box = boxplot(col)
    box.update_layout(width=int(450))
    outliers = dataframe.outliers(col)
    minoutliers = outliers['Min outliers']
    maxoutliers = outliers['Max outliers']
    st.markdown("Total number of outliers: " + str(len(maxoutliers)))
    st.markdown("Total number of unique outliers: " + str(len(np.unique(list(maxoutliers.values())))))
    col1, col2, col3, col4 = st.columns((1, 1, 1, 3))
    col1.dataframe({"Min outliers":minoutliers}, width=150)
    col2.dataframe({"Max outliers":maxoutliers}, width=150)
    col3.dataframe({"Unique outliers":np.unique(list(maxoutliers.values()) + list(minoutliers.values()))}, width=150)
    col4.plotly_chart(box)


dataframe.missing_values_imputation()
dataframe.encode_str()

col = st.sidebar.selectbox(
    "Attribute",
    list(data.keys()))

plot_col(col) 
