
from features.feature_extractor import Features
from scipy.io.matlab.miobase import (MatFileReader, docfiller, matdims, read_dtype,
                                     arr_to_chars, arr_dtype_number, MatWriteError,
                                     MatReadError, MatReadWarning)
from features import feature_extractor
from features import imagefeatures
import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
from keras import Sequential, optimizers
from keras.layers import Input, Dense, Dropout, LSTM, RepeatVector, TimeDistributed, BatchNormalization
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from auto_encoder import encoderdecoder
import streamlit as st
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from keras.models import load_model
import pandas as pd
import re
from scipy.io import loadmat
import numpy as np
import sys
from sklearn.decomposition import PCA
import pickle as pkl
from preprocessing import specConvert
from PIL import Image
from ECGworkflow import ecg_workflow
from PCGworkflow import pcg_workflow
from PCGSpect_workflow import pcg_spec_workflow
from PCGSpec_Prediction import prediction_8_cluster
import os 



## READING THE MAT FILES
def read_mat_files(file_path):
                df_list = []
                mat = loadmat(file_path)
                print(mat['val'][0] , ' | file: ' , file_path)
                x = file_path.split('.')[0]
                data = mat['val'][0]
                data = np.append(x, data)
                df_list.append(data)
                df = pd.DataFrame(df_list)
                return df

def r2workflow_lstmae(waveform_path):
    df_list = read_mat_files(waveform_path)
    df_list = df_list.drop(0, axis = 1)
    ## CHOPPING THE FILE TO 2500 POINTS
    df_list = df_list.iloc[:, :2500]
                

    ## DISPLAYING THE READ FILE (TRIMMED)
    st.write("Reading the ECG File:")
    st.dataframe(df_list)
    def reshape_data(X_train):
        X_train = X_train.reshape(X_train.shape[0], 1,  X_train.shape[1])
        return X_train
    df_list = df_list.astype(float)
    dd = reshape_data(df_list.to_numpy())
    print("106: ", dd.shape)
    print(dd)
    print(type(dd))
    model = load_model('./model_folder/LSTM-AE-2016-2500-86-92.h5')
    optimizer = Adam(learning_rate=0.001)
    loss = 'mse'
    encoder_model = Model(inputs=model.inputs, outputs=model.get_layer(index=5).output)
    optimizer = Adam(learning_rate=0.001)
    loss = 'mse'
    encoder_model.compile(optimizer=optimizer, loss=loss)

    latent_representation = encoder_model.predict(dd)
    print("123: LR", latent_representation)
    pca_reload = pkl.load(open("./model_folder/LSTM_Best_92PCA.pkl",'rb'))
    result_new = pca_reload.transform(latent_representation) 
    X_PCA=pd.DataFrame(result_new)
    X = pd.DataFrame(latent_representation)
    FINAL_X_SET = pd.concat([X, X_PCA], axis=1, join='inner')
    with open('./model_folder/LSTM_Best_92CLF.pkl', 'rb') as f:
        model = pkl.load(f)
    res = model.predict(FINAL_X_SET)
    y = res

    st.write("Newly Constructed Features using Encoder:")
    st.dataframe(X)

    st.text("-- PREDICTED RESULT FOR FILE " + waveform_path + " -- ")
    print(y)
    if(y==0):
        st.write("SUBJECT ECG ABNORMAL")
    elif(y==1):
        st.write("SUBJECT ECG NORMAL")

    st.text("-- ACTUAL RESULT FOR FILE " + waveform_path + " -- ")

    # GETTING THE ACTUAL PREDICTIONS
    df_actual = pd.read_csv("2016_17_ALL_ECG_SUBJECTS_WITH_LABEL.csv")
    x = waveform_path
    # MODIFY [5] BASED ON THE FUNCTION FILE_SELECTOR'S FOLDER PATH !!
    x = x.split(".")[1].split(".")[0].split("/")[-1]
    print("X IS ")
    print(x)
    x = str(x)
    z = df_actual[df_actual['file_name'].str.contains(x)]
    #print(df_actual['file_name']+str("IS THE PATIENT!"))
    m = z['label']
    m = np.array(m)
    if(m == [1]):
        print("patient normal")
        st.text("SUBJECT ECG NORMAL")
    elif(m == [-1]):
        print("patient abnormal")
        st.text("SUBJECT ECG ABNORMAL")
    else:
        # ground truth not provided
        st.text("Ground Truth Inconclusive Result (~)")