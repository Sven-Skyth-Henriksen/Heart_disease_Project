import streamlit as st
import pandas as pd
from PIL import Image
import base64


def app():

    media = st.beta_container()
    image = st.beta_container()
    model = st.beta_container()
    
        
    with image:
        image = Image.open('im3.jpeg')
        st.image(image , caption='')
        
        
    with model:
        st.markdown('Here you can download our model:')
        st.markdown('***[Model](https://github.com/Sven-Skyth-Henriksen/Heart_disease_Project)***')
        
        
           