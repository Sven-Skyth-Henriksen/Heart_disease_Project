import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px


def app():

    media = st.beta_container()
    image = st.beta_container()

    with media:
        # Another1

        with st.beta_expander('• The person who are suffering the most.'):
            st.subheader('Variation of Age for each target class')
            image = Image.open('im1.jpeg')
            st.image(image , caption='')
            st.markdown(''' 
            • Target = 1 implies that the person is suffering from heart disease\n
            • Target = 0 implies the person is not suffering.\n
            • That most people who are suffering are of the age of 58, followed by 57.\n
            • Majorly, people belonging to the age group 50+ are suffering from the disease\n
            ''')

            

        # Another1
        with st.beta_expander('• Distribution of age vs sex.'):
            st.subheader('Distribution of age vs sex with the target class')
            image = Image.open('im2.jpeg')
            st.image(image , caption='')
            st.markdown(''' 
            • 0 is female and 1 is male.\n
            • Females who are suffering from the disease are older than males.\n
            ''')
            
            
              # Another1
        with st.beta_expander('• Correlation of the Data'):
            image = Image.open('correlation.jpeg')
            st.image(image , caption='')
            




app()


