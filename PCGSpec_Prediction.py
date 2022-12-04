# from keras.models import load_model
#from tensorflow.keras.preprocessing import image
import streamlit as st
import pandas as pd
import numpy as np



def prediction_8_cluster(path, waveform_path):
        # code for prediction

        print("entered pcgspec_prediction file")
# --------------------------- ADD THE MODEL PREDICTION CODE HERE WITH h5 FILE + i added the code snippet temporarily, feel free to delete ------
        #img1 = image.load_img(path)
        # model = load_model('my_model.h5')
        # img_array = image.img_to_array(img)

        # img_batch = np.expand_dims(img_array, axis=0)
        # prediction = model.predict(img1_batch)


# ------------ temp value PLS CHANGE ONCE ADDING THE MODEL ------------------------------
        prediction = 1

# ------------------ DISPLAYING THE RESULTS ON UI ---------------------------------
        # st.write("PREDICTION : "+str(val))
        st.text("-- PREDICTED RESULT FOR FILE " + waveform_path + " -- ")
        if(prediction == [-1]):
            st.text("SUBJECT PCG NORMAL")
        else:
            st.text("SUBJECT PCG ABNORMAL")



        st.text("-- ACTUAL RESULT FOR FILE " + waveform_path + " -- ")
        df_actual = pd.read_csv("REFERENCE.csv")
        x = waveform_path
 
        x = x.split(".")[1].split(".")[0].split("/")[-1]
        print("X IS ")
        print(x)
        m = 2

        z = df_actual[df_actual['file_name'].str.contains(x)]
        m = z['label']
        m = np.array(m)
        if(m == [-1]):
            print("patient normal")
            st.text("SUBJECT PCG NORMAL")
        elif(m == [1]):
            print("patient abnormal")
            st.text("SUBJECT PCG ABNORMAL")
        else:
            st.text("Actual Record of Patient -> No Ground Truth")
    