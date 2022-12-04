import streamlit as st
import pandas as pd
import numpy as np
# from tensorflow.keras.optimizers import Adam
import keras
from keras.models import load_model
import cv2


def prediction_8_cluster(path, waveform_path):

        print("entered pcgspec_prediction file")
        model=load_model('DENSENET.h5')
        #optimizer = Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.9997, epsilon=0.0001, decay=0.0)
        optimizer = 'adam'
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        img=cv2.imread(path)
        img=cv2.resize(img,(64,64))
        img=np.reshape(img,[1,64,64,3])
        classes=model.predict(img)

        prediction=np.argmax(classes,axis=1)
        print("PREDICTION FOR DL:",prediction)

        st.text("-- PREDICTED RESULT FOR FILE " + waveform_path + " -- ")
        if(prediction == [0]):
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
        # ground truth not provided
            st.text("Ground Truth Inconclusive Result (~)")
    