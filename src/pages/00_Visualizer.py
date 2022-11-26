from DataFrame import DF
import streamlit as st  

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def get_dataframe():
    return DF()

data = get_dataframe()

row = st.sidebar.number_input('Row', max_value=100, value=1)
col = st.sidebar.selectbox(
    "Attribute",
    data.data.keys())
value = st.sidebar.text_input('Value', value="0")

if st.sidebar.button('Update'):
    if (str.isdigit(value)):
        data.data.get(col)[row] = int(value)
    else: 
        data.data.get(col)[row] = value

if st.sidebar.button('Fill Missing'):
    data.missing_values_imputation()
    
#if st.sidebar.button('Encode'): 
#    data.encode_str() 
    
st.subheader("Database")
st.dataframe(data.data)
#st.subheader("Details") 

#try:
#    summary = {k: data.summary(k) for k in list(data.data.keys())}
#    st.dataframe(summary) 

#except:
#    st.markdown('You need to encode first')