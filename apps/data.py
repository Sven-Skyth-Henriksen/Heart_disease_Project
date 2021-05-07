import streamlit as st
import pandas as pd
from PIL import Image


def app():
    
    data = st.beta_container()
    
    dx = pd.read_csv('heart.csv')
    

    with data:
        st.write("""In this project, we navigate through Heart data to find out if a person has a heart disease or not. From the data, 
     we got some interesting insights that you might be curious to know and stored the answers in a readable format for your convinience.""")
        st.markdown('Here you can find the source code : [Heart_disease_Project](https://github.com/Sven-Skyth-Henriksen/Heart_disease_Project)')
        st.title('Data')
        st.markdown("![Data](https://media3.giphy.com/media/4FQMuOKR6zQRO/giphy.webp?cid=ecf05e47teulwamthzfh5rqgji0j4nmfxw1b52c2rgsybk50&rid=giphy.webp&ct=g)")

        if st.checkbox('Reveal the data'):
            st.subheader('Heart data')
            st.write(dx) #.header(50) inside the the ()
