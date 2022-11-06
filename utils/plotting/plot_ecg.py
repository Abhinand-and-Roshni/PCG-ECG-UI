from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import matplotlib as mpl


def img(waveform_path):
    mpl.rcParams['figure.figsize'] = [20, 10]
    print('rcparams set')
    mat = scipy.io.loadmat(waveform_path)
    data = np.array(mat['val'])
    data = np.transpose(data)
    list = []
    for x in data:
        for i in x:
            list.append(i)
        data = list
    data = np.array(data)
    fig, ax = plt.subplots()
    print('subplots created')
    ax.plot(data)
    print('plotted')
    fig = ax.get_figure()
    ax.title.set_text("ECG Data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Voltage")
    fig.savefig("ecg_signal_read.png")
    print('saved figure')
    image = Image.open('ecg_signal_read.png')
    return image
