import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
import streamlit.components.v1 as components
import codecs
from sklearn.datasets import load_boston


#------------------------
# here we are going to set the page layout
st.set_page_config(page_title='Exploratory data analysis app',layout='wide')


m = st.markdown("""
<style>
div.stButton > button:first-child {

     background-color: white;
     color: black;
     border: 2px solid #4CAF50; /* Green */
    
}
div.stButton > button:hover {
     background-color: #4CAF50; /* Green */
     color: white;
}

[theme]
base="light"
primaryColor="#7d7878"
backgroundColor="#e0dddf"
secondaryBackgroundColor="#d2d0d1"
</style>""", unsafe_allow_html=True)


st.write("""
# The Exploratory data analysis app

This web application will help you in your initial exploratory data analysis. This applciation will help you automate your **EDA** very eaily.
Click on the import button on the side bar to import your dataset. This app will also help you to generate automated EDA reports using SweetViz and Pandas Profiling.

By **NisAr PakhtoOn**

""")

data = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

if data is not None:
    data = pd.read_csv(data)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(data)
    st.write('Numeric Columns :')
    numerics = ['int16', 'int32', 'int64']
    df = data.select_dtypes(include=numerics)
    st.info(df.columns)
    df1 = data.select_dtypes(exclude=["number","bool_","object_"])
    st.write('Categorical columns :')
    st.info(df1.columns)

else:

    boston = load_boston()
    data = pd.DataFrame(boston.data)
    data.columns = boston.feature_names

    st.markdown('The Boston housing dataset is used as the example.')
    st.write(data.head(5))


if st.button('Show Statistics'):
     st.write('The statisitical analysis of your data is shown below')
     st.write(data.describe().style.background_gradient(cmap='pink_r'))

if st.button('Show Null values'):
     st.write('Null values are shown below')
     st.write(data.isnull().sum())

if st.button('Show skewness'):
     st.write('Skewness is shown below')
     st.write(data.skew())
     numeric_data = data._get_numeric_data()
     for i in numeric_data.columns:
        fig = plt.figure(figsize=(10, 4))
        sns.distplot(data[i])
        st.write(fig)

if st.button('Find Outliers'):
     st.write('Outliers')
     numeric_data = data._get_numeric_data()
    #  z = np.abs(stats.zscore(data))
    #  st.write(z)
     Q1 = np.percentile(data, 25,
                   interpolation = 'midpoint')
 
     Q3 = np.percentile(data, 75,
                   interpolation = 'midpoint')
     st.write(Q1)
     st.write(Q3)

if st.button('Visualize'):
     st.write('Data Visualization!')
     st.write(data.corr())
     numeric_data = data._get_numeric_data()
     
     fig, ax = plt.subplots()
     corr = data.corr()
     sns.set(rc = {'figure.figsize':(20,15)})
     sns.heatmap(data.corr(), ax=ax,annot=True)
     st.write(fig)
     st.bar_chart(data)
     st.line_chart(data)
     st.area_chart(data)

if st.button('Profile Report'):
    st.write('Pandas Profiling Report :')
    report = data.profile_report()
    st_profile_report(report)

def st_display_sweetviz(report_html,width=800,height=1000):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)

if st.button('SweetViz Report'):
    st.write('SweetViz EDA Report')
    sweetreport = sv.analyze(data)
    sweetreport.show_html()
    st_display_sweetviz('SWEETVIZ_REPORT.html')
