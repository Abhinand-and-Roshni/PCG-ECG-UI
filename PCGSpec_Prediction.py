# from keras.models import load_model
#from tensorflow.keras.preprocessing import image
import streamlit as st
import pandas as pd
import numpy as np
# returns a compiled model
# identical to the previous one



def prediction_8_cluster(path, waveform_path):
        # code for prediction

        print("entered pcgspec_prediction file")
        #img1 = image.load_img(path)
        # model = load_model('my_model.h5')
        # img_array = image.img_to_array(img)

        # img_batch = np.expand_dims(img_array, axis=0)
        # prediction = model.predict(img1_batch)

        prediction = 1

        # st.write("PREDICTION : "+str(val))
        st.text("-- PREDICTED RESULT FOR FILE " + waveform_path + " -- ")
        if(prediction == [-1]):
            st.text("SUBJECT PCG NORMAL")
        else:
            st.text("SUBJECT PCG ABNORMAL")
        st.text("-- ACTUAL RESULT FOR FILE " + waveform_path + " -- ")
        df_actual = pd.read_csv("REFERENCE.csv")
        x = waveform_path
        #x = x.split(".")[0].split("/")[-1]
        x = x.split(".")[1].split(".")[0].split("/")[-1]
        print("X IS ")
        print(x)
        m = 2
        # for i in range(408):
        #     if df_actual.iloc[i, 0] == x:
        #         print('x is found!!!!!')
        #         m = df_actual.iloc[i, 1]
        # print(m)
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
    