import radiomics
import pandas as pd
import re
import imageio as iio


def getImageFeatures():
    img = iio.imread(
        './spectrograms/Spectrogram.png')
    xyz = radiomics.features.group1_features(img)
    output1 = pd.DataFrame()
    output1 = output1.append(xyz, ignore_index=True)
    print('statistical features extracted!')
    pqr = radiomics.gray_level_cooccurrence_features(img, img)
    output2 = pd.DataFrame()
    output2 = output2.append(pqr, ignore_index=True)
    output2.rename(columns={'variance': 'variance.1'}, inplace=True)
    print('glc features extracted!')
    final = pd.concat([output1, output2], axis=1)
    return final
