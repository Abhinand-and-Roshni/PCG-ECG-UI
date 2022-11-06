from features.feature_extractor import Features
from features import feature_extractor
from features import imagefeatures
from scipy.io.matlab.miobase import (MatFileReader, docfiller, matdims, read_dtype,
                                     arr_to_chars, arr_dtype_number, MatWriteError,
                                     MatReadError, MatReadWarning)
import streamlit as st
import pandas as pd
import re
import os
import sys
import numpy as np
from sklearn.decomposition import PCA
import pickle as pkl
from preprocessing import specConvert
from PIL import Image
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
import requests  # pip install requests
from utils.plotting import plot_ecg


def ecg_workflow(waveform_path):
    try:
        image = plot_ecg.img(waveform_path)
        st.image(image)
    except MatReadError:
        st.error('Enter ECG data in form of MAT file.')

    labels = [1, -1]
    fs = 300
    ecg_features = Features(file_path=waveform_path, fs=fs,
                            feature_groups=['full_waveform_features'])

    # Calculate ECG features
    ecg_features.extract_features(
        filter_bandwidth=[3, 45], n_signals=None, show=True,
        labels=labels, normalize=True, polarity_check=True,
        template_before=0.25, template_after=0.4
    )

    dic_fwff = feature_extractor.features

    rri_features = Features(file_path=waveform_path, fs=fs,
                            feature_groups=['rri_features'])
    print(waveform_path)
    # Calculate ECG features
    rri_features.extract_features(
        filter_bandwidth=[3, 45], n_signals=None, show=True,
        labels=labels, normalize=True, polarity_check=True,
        template_before=0.25, template_after=0.4
    )

    dic_rri = feature_extractor.features
    dic1 = [dic_rri]
    # st.write(dic1)
    df11 = pd.DataFrame(dic1)
    df11 = df11.iloc[:, 1:]
    st.dataframe(df11)
    df11_list = df11.values

    pca_1 = pkl.load(open("pca16_ecg_2017.pkl", 'rb'))
    df111 = pca_1.transform(df11)

    fin_df = np.concatenate((df11, df111), 1)

    pickled_model = pkl.load(open("RFC_Boosting_ECG17.pkl", "rb"))
    prediction = pickled_model.predict(fin_df)
    st.text("-- PREDICTED RESULT FOR FILE " + waveform_path + " -- ")
    if(prediction == 1):
        st.text("SUBJECT ECG NORMAL")
    else:
        st.text("SUBJECT ECG ABNORMAL")
    st.text("-- ACTUAL RESULT FOR FILE " + waveform_path + " -- ")
    df_actual = pd.read_csv("2016_17_ALL_ECG_SUBJECTS_WITH_LABEL.csv")
    x = waveform_path
    # MODIFY [5] BASED ON THE FUNCTION FILE_SELECTOR'S FOLDER PATH !!
    x = x.split(".")[0].split("/")[-1]
    print("X IS ")
    print(x)
    z = df_actual[df_actual['file_name'].str.contains(x)]
    print(df_actual['file_name']+str("IS THE PATIENT!"))
    m = z['label']
    m = np.array(m)
    if(m == "N" or m == "1"):
        print("patient normal")
        st.text("SUBJECT ECG NORMAL")
    elif(m == "O" or m == "A" or m == "-1"):
        print("patient abnormal")
        st.text("SUBJECT ECG ABNORMAL")
    else:
        st.text("Actual Record of Patient -> No Ground Truth")
