from PIL import Image
import streamlit as st


def app():
    header = st.beta_container()
    image = st.beta_container()

    with header:
        st.title(' Predict if a patient has a Heart disease. ')     
        st.text('')
        st.text('')
        st.text('')
        st.text('')
        st.title('⚜️ About ⚜️:')
        st.markdown('''
        
    
    We think that the with help of our data we can figure out if a patient has a heart disease or not. For this we are using Machine Learning and Feature Engineering.
    
    
   ⬇️⬇️ This Project was created by 3 ***[Strive School](https://strive.school/)*** Students.⬇️⬇️
   
    ''')
    
        st.markdown('## Get to know us 👋🏻:')

    
        if st.button('Click here'):
            st.balloons()
            st.markdown("![Hello There](https://media1.giphy.com/media/i4MAH84pqe2m2aVojc/200w.webp?cid=ecf05e479ijit106tasvpbdg5q3t9xp548pxxrbj9iaa9qvb&rid=200w.webp&ct=g)")
            st.write('***The Developer Team***:')
            st.write('• Deniz Elci: [GitHub ](https://github.com/Deniz-shelby)&[ LinkedIn](https://www.linkedin.com/in/deniz-elci-2500b2205/)')
            st.write('• Sven Skyth Henriksen: [GitHub ](https://github.com/Sven-Skyth-Henriksen)&[ LinkedIn](https://www.linkedin.com/in/sven-skyth-henriksen-4857bb1a2/)')
            st.write('• Stephen George: [GitHub ](https://github.com/stephengeorge93)&[ LinkedIn](https://www.linkedin.com/in/stephen-george-b79941116/)')
            
        
    with image:
        image = Image.open('team.jpeg')
        st.image(image , caption='')