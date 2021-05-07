from PIL import Image
import streamlit as st
from multiapp import MultiApp

from apps import home, about, data, graph, graph2, result # importing the apps here

app = MultiApp()



st.markdown("![title gif](https://media1.giphy.com/media/3ohhwIMxSkDTcC2r6g/200w.webp?cid=ecf05e47w9vnuwnk80p1vzh7sc52rd9eqmofqq5dgqpzurpi&rid=200w.webp&ct=g)")


st.markdown(''' ## Welcome to our website''')
st.text(' ')
st.markdown('----------------------------------------------- ')
st.text(' ')
st.text(' ')

st.markdown(' ')
st.markdown('##  MAY THE CODE BE WITH YOU   ')
st.markdown('----------------------------------------------- ')
st.markdown(' ')
st.markdown('Please select a page:')

# Add all the application here
app.add_app('Home', home.app)
#app.add_app("About", about.app)
#app.add_app("Data", data.app)
#app.add_app('Galaxies Analysis', graph.app)
#app.add_app('Planets Analysis', graph2.app)
#app.add_app('Prologue', result.app)



# The main app
app.run()