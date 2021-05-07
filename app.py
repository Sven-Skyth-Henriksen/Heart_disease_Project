from PIL import Image
import streamlit as st
from multiapp import MultiApp

from apps import home, about, data, data_v, model# importing the apps here

app = MultiApp()



st.markdown("![title gif](https://media3.giphy.com/media/FoVzfcqCDSb7zCynOp/200w.webp?cid=ecf05e47yvu34geal6scw0t3ejyzo8e3dvlufof2vxhii7m2&rid=200w.webp&ct=g)")


st.markdown(''' ## Welcome to our website''')
st.text(' ')
st.markdown('----------------------------------------------- ')
st.text(' ')
st.text(' ')

st.markdown(' ')
st.markdown("##  Good health and good sense are two of life's greatest blessings.")
st.markdown('----------------------------------------------- ')
st.markdown(' ')
st.markdown('Please select a page:')

# Add all the application here
app.add_app('Home', home.app)
app.add_app("About", about.app)
app.add_app("Data", data.app)
app.add_app('Data Analysis', data_v.app)
app.add_app('Model', model.app)
#app.add_app('Prologue', result.app)



# The main app
app.run()