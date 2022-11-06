from features.feature_extractor import Features
from features import feature_extractor
from features import imagefeatures
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
from preprocessing import LOESS


def pcg_workflow(waveform_path):
    LOESS.loess_filtering(waveform_path)
    specConvert.getSpectrogram(waveform_path)
    image = Image.open(
        'C:/Users/Roshni/OneDrive/Desktop/Sem7/Project 1/ECG UI/spectrograms/Spectrogram.png')
    st.image(image, caption='Mel Spectrogram')
    imageFeaturesExtracted = imagefeatures.getImageFeatures()
    st.write('Image features - ')
    st.dataframe(imageFeaturesExtracted)
    KEY_TO_CHECK = re.findall(".\d{4}", waveform_path)[0]
    liszt = []
    datafr = pd.read_csv(
        "C:/Users/Roshni/OneDrive/Desktop/Sem7/Project 1/ECG UI/PHYSIONET_NORMALIZED_AND_SCALED_DATASET (2).csv")
    datatocheck = pd.read_csv(
        'C:/Users/Roshni/OneDrive/Desktop/Sem7/Project 1/ECG UI/PHYSIONET_NORMALIZED_AND_SCALED_DATASET (2).csv')
    DA = datatocheck[['patientID', 'OccupiedBandwidth', 'MFCC8', 'MFCC9', 'MFCC12', 'MaxtoMinDifference', 'MFCC1', 'MFCC4', 'MFCC11', 'MFCC15', 'MFCC16', 'MFCC10', 'MFCC7', 'MFCC18', 'MFCC17', 'PeakMagToRMS', 'MFCC3', 'MFCC14',
                      'MFCC13', 'MeanFrequency', 'MFCC5', 'BandPower', 'MFCC2', 'MFCC6', 'MFCC19', 'MFCC22', 'MFCC23', 'MedianFrequency', 'MFCC21', 'MFCC20', 'MFCC24', 'MFCC25', 'MFCC26', 'ENB-NORM', 'MFCC27', 'ZeroCrossRate', 'MFCC28']]
    results = 0
    for i in range(3000):
        if DA.iloc[i, 0] == KEY_TO_CHECK:
            print('FOUND IN DATASET')
            waveFeatureMatrix = datafr.iloc[i, 4:]
            st.write('Wave features - ')
            waveFeatureMatrix = pd.DataFrame(data=waveFeatureMatrix)
            waveFeatureMatrix = waveFeatureMatrix.transpose()
            print('transposed!')
            st.dataframe(waveFeatureMatrix)
    X = np.concatenate((imageFeaturesExtracted, waveFeatureMatrix), 1)
    pca_1 = pkl.load(open("PCG-PCA-Pickle.pkl", 'rb'))
    pc_X = pca_1.transform(X)
    finalX = np.concatenate((X, pc_X), 1)
    pickled_model = pkl.load(open("bestPCGmodel.pkl", 'rb'))
    prediction = pickled_model.predict(finalX)
    st.text("-- PREDICTED RESULT FOR FILE " + waveform_path + " -- ")
    if(prediction == 1):
        st.text("SUBJECT PCG NORMAL")
    else:
        st.text("SUBJECT PCG ABNORMAL")
    st.text("-- ACTUAL RESULT FOR FILE " + waveform_path + " -- ")
    df_actual = pd.read_csv("REFERENCE.csv")
    x = waveform_path
    x = x.split(".")[0].split("/")[-1]
    m = 2
    for i in range(408):
        if df_actual.iloc[i, 0] == x:
            print('x is found!!!!!')
            m = df_actual.iloc[i, 1]
    if(m == 1):
        print("patient normal")
        st.text("SUBJECT PCG NORMAL")
    elif(m == -1):
        print("patient abnormal")
        st.text("SUBJECT PCG ABNORMAL")
    else:
        st.text("Actual Record of Patient -> No Ground Truth")
