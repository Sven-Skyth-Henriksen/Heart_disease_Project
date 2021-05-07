mport streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px


def app():

    media = st.beta_container()
    model = st.beta_container()

    with media:
        st.markdown('## Feel free to dowland the Model here:')
        
    with model:
        uploaded_files = st.file_uploader("Hippocratia_Heart_Monitoring.ipynb", accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("Hippocratia_Heart_Monitoring:", uploaded_file.name)
            st.write(bytes_data)
        