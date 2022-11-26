import streamlit as st  
import plotly.graph_objs as go
from DataFrame import DF

st.set_page_config(layout="wide")

st.subheader("Correlation")

df = DF()
df.missing_values_imputation()
df.encode_str()
data = df.data

att1 = st.sidebar.selectbox(
    "Attribute 1",
    data.keys())

att2 = st.sidebar.selectbox(
    "Attribute 2",
    data.keys()) 

def scatter(var_select1, var_select2):
    tmp1 = list(data.get(var_select1))
    tmp2 = list(data.get(var_select2))

    tmp1 = [df.encode_dict.get(var_select1)[i] for i in tmp1] if var_select1 in df.encode_dict.keys() else tmp1
    tmp2 = [df.encode_dict.get(var_select2)[i] for i in tmp2] if var_select2 in df.encode_dict.keys() else tmp2

    trace = go.Scatter(
        x=tmp1,
        y=tmp2,
        opacity=0.7, marker_color='gold')
    
    layout = dict(xaxis=dict(title= var_select1), yaxis=dict(title= var_select2))
    fig = go.Figure(data=[trace], layout=layout) 
    fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    return fig 

st.plotly_chart(scatter(att1, att2)) 