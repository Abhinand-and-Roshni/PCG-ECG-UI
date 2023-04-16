
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
from tensorflow.keras.models import Model
from keras import Sequential, optimizers
from keras.layers import Input, Dense, Dropout, LSTM, RepeatVector, TimeDistributed, BatchNormalization
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from auto_encoder import encoderdecoder
import streamlit as st
import torch
from tensorflow.keras import layers, losses
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
from rev2ECGWorkflow import r2workflow_lstmae

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

def r1ecg_lr_vanillaAE(waveform_path1):
    df_list = read_mat_files(waveform_path1)
    
    df_list = df_list.drop(0, axis = 1)
    df_list = df_list.iloc[:, :3600] # SPECIFIC FOR THIS MODEL BC TRAINED FOR 3600 POINTS (COMMON)

    st.write("Reading the ECG File:")
    st.dataframe(df_list)
    df_list = df_list.astype(float)
    wave_tf = tf.convert_to_tensor(df_list)
    wave_tf = tf.cast(wave_tf, dtype=tf.float32)

    # Recreate the model
    model = encoderdecoder.detector()

    model.compile(optimizer='adam', loss='mae')
    model.build(input_shape=(None, 3600))

    # abhi path!!
    # model.load_weights("/users/abhinandganesh/Downloads/autoencoder_weights.h5")

    # DONT DELETE !! ROSHNI PATH
    model.load_weights('C:/Users/Uma Bala/OneDrive/Desktop/Sem7/Project-II/February/autoencoder_weights.h5')

    x = model.encoder(wave_tf)
    print(x)
    xx1 = pd.DataFrame(x.numpy())

    st.write("Newly Constructed Features using Encoder:")
    st.dataframe(xx1)

    with open("./model_folder/nb_weights.pkl", "rb") as f:
        weights1 = pkl.load(f)

    xx = xx1 * weights1 * weights1
    # print("line 91: ")
    # print(xx)
    # print("line 92: ")
    # print(xx1)

    with open( "./model_folder/LR_Bayes_16.pkl", "rb" ) as f:
        LRB = pkl.load(f)
    y = LRB.predict(xx)
    st.text("-- PREDICTED RESULT FOR FILE " + waveform_path1 + " -- ")
    print(y)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    if(y==-1):
        st.write("SUBJECT ECG ABNORMAL")
    elif(y==1):
        st.write("SUBJECT ECG NORMAL")

    st.text("-- ACTUAL RESULT FOR FILE " + waveform_path1 + " -- ")
    df_actual = pd.read_csv("2016_17_ALL_ECG_SUBJECTS_WITH_LABEL.csv")
    x = waveform_path1
    print("TESTTTTTTTTT"+waveform_path1)
    # MODIFY [5] BASED ON THE FUNCTION FILE_SELECTOR'S FOLDER PATH !!
    x = x.split(".")[1].split(".")[0].split("/")[-1]
    print("X IS ")
    print(x)
    x = str(x)
    z = df_actual[df_actual['file_name'].str.contains(x)]
    #print(df_actual['file_name']+str("IS THE PATIENT!"))
    m = z['label']
    m = np.array(m)
    #print("*****87:", m)
    if(m == [1]):
        print("patient normal")
        st.text("SUBJECT ECG NORMAL")
    elif(m == [-1]):
        print("patient abnormal")
        st.text("SUBJECT ECG ABNORMAL")
    else:
        # ground truth not provided
        st.text("Ground Truth Inconclusive Result (~)")