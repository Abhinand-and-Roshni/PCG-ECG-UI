import streamlit as st
import pandas as pd
import re
import os
import sys
import numpy as np
from sklearn.decomposition import PCA
import pickle as pkl
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from maad import sound, rois, features
from maad.util import (power2dB, plot2d, format_features, read_audacity_annot,
                       overlay_rois, overlay_centroid)

def getRegions(waveform_path):
    s, fs = sound.load(waveform_path)
    dB_max = 96
    Sxx_power, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=1024//2)
    Sxx_db = power2dB(Sxx_power) + dB_max
    Sxx_power_noNoise= sound.median_equalizer(Sxx_power, display=False, **{'extent':ext})
    Sxx_db_noNoise = power2dB(Sxx_power_noNoise)
    Sxx_db_noNoise_smooth = sound.smooth(Sxx_db_noNoise, std=0.5,
                         display=False, savefig=None,
                         **{'vmin':0, 'vmax':dB_max, 'extent':ext})
    im_mask = rois.create_mask(im=Sxx_db_noNoise_smooth, mode_bin ='relative',
                           bin_std=8, bin_per=0.5,
                           verbose=False, display=False)
    im_rois, df_rois = rois.select_rois(im_mask, min_roi=25, max_roi=None,
                                 display= False,
                                 **{'extent':ext})
    df_rois = format_features(df_rois, tn, fn)
    ax0, fig0 = overlay_rois(Sxx_db, df_rois, **{'vmin':0, 'vmax':dB_max, 'extent':ext})
    fig0.set_figheight(15)
    fig0.set_figwidth(15)

    df_centroid = features.centroid_features(Sxx_db, df_rois, im_rois)
    df_centroid = format_features(df_centroid, tn, fn)
    ax0, fig0 = overlay_centroid(Sxx_db, df_centroid, savefig=None,
                             **{'vmin':0,'vmax':dB_max,'extent':ext,'ms':4,
                                'marker':'+', 'fig':fig0, 'ax':ax0})    
    fig0.savefig('./img_ui/RegionsofInterest.png')

