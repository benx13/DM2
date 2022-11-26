import streamlit as st  
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff
from DataFrame import DF

st.set_page_config(layout="wide")

dataframe = DF()
dataframe.missing_values_imputation()
dataframe.encode_str()

data = dataframe.data
df = dataframe.df


def plot_distribution_1(var_select): 
    colors = ['#FFD700', '#7EC0EE']

    fig = ff.create_distplot([list(dataframe.sorted(var_select))], [''], colors = colors, show_hist = False, curve_type='normal')

    fig.update_layout(showlegend=False)
    return fig

def barplot(var_select, nbinsx=50, freq=False, width=False): 

    if freq:
        col = dataframe.reduce_f(var_select, nbinsx)
        trace1 = go.Histogram(
            x=col,
            name='', 
            marker=dict(
                color='gold', 
                line=dict(color='#000000',width=1)),
            marker_color='gold', 
            opacity=0.8,
            bingroup=1
        )
    else:
        if width:
            col = dataframe.reduce_w(var_select, nbinsx)
        else:
            col = data.get(var_select)
            col = sorted([dataframe.encode_dict.get(var_select)[i] for i in col] if var_select in dataframe.encode_dict.keys() else col)

        trace1 = go.Histogram(
            x=col,
            autobinx=False, 
            name='', 
            marker=dict(
                color='gold', 
                line=dict(color='#000000',width=1)),
            marker_color='gold', 
            opacity=0.8,
            bingroup=1
        )
    
    layout = dict(xaxis=dict(title = var_select), yaxis=dict(title= 'Count'), showlegend=False)
    fig = go.Figure(data=[trace1])
    if (not width and not freq):
        fig.add_vline(x=dataframe.Q(1, var_select), name='Q1', line_width=3, line_dash="dash", line_color="lightskyblue")
        fig.add_vline(x=dataframe.Q(2, var_select), name='Q2', line_width=3, line_dash="dash", line_color="lightskyblue")
        fig.add_vline(x=dataframe.Q(3, var_select), name='Q3', line_width=3, line_dash="dash", line_color="lightskyblue")
    fig.update_layout(layout) 
    return fig

def boxplot(var_select): 
    trace = go.Box(
        boxpoints='outliers',
        y=list(dataframe.data.get(var_select)),
        name=var_select, opacity = 0.8, marker=dict(
        color='gold', line=dict(color='#000000',width=1)),
)

    
    layout = dict(xaxis=dict(), yaxis=dict(title= 'Count'))
    fig = go.Figure(data=[trace], layout=layout)
    
    return fig



col = st.sidebar.selectbox(
    "Attribute",
    list(data.keys()))

def plot_col(col, bin=50, freq=False, width=False):
    st.subheader("Plots")
    
    st.markdown("Bar plot")

    bar = barplot(col, bin, freq=freq, width=width)
    bar.update_layout(width=int(700))


    plot = plot_distribution_1(col)
    plot.update_layout(width=int(700))

    st.plotly_chart(bar)

    st.plotly_chart(plot) 

if st.sidebar.button('Clean'):
    dataframe.outliers_median_imputation(col, dataframe.outliers(col)) 

st.sidebar.markdown('Normalize')
col2, col3, col4 = st.sidebar.columns([1.5, 1.5, 1]) 

if col2.button('MinMax'):
    dataframe.normalize_minmax(col)

if col3.button('MeanStd'):
    dataframe.normalize_meanstd(col)

if col4.button('OG'): 
    pass

st.sidebar.markdown('Reduce')
type = st.sidebar.radio(
    "Select",
    ('Width', 'Frequency'))

col1, col2 = st.sidebar.columns([2, 1]) 

bin = col1.number_input('Bin Size', value=0)
col2.markdown('')
if col2.button('Reduce'):
    if (type == 'Width'):
        plot_col(col, bin, width=True)
    else:
        plot_col(col, bin, freq=True)

else:
    plot_col(col)