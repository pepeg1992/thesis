from funzioniSupplementari import getPhi
import numpy as np
from constants import *


##############      SCEGLIERE IL TIPO DI CONTESTO, DATASET E STANZA     ##############
simulatedcheck=False
realcheck=False
hsc=True
fittizio=False
context=15
numberdir_fittizio=1000
nameplace='Kitchen'
dir_training='_'
if simulatedcheck:
    dir_training=dir_training+'evalita_'
if hsc:
    dir_training=dir_training+'hscma_'
if fittizio:
    dir_training=dir_training+'dls_'
dirAudio2='/Signals/Mixed_Sources/'+nameplace
if nameplace=='Kitchen':
    pathdir_fittizio=pathdir_fittizio+'DIRHALibriSpeechsource_083seconds_reverberation/'
    mic_log_mel=13 #numero di microfoni nella cucina
else:
    mic_log_mel=15
    pathdir_fittizio=pathdir_fittizio+'DIRHALibriSpeechsource_074seconds_reverberation/'
window=np.hanning(N)
lun=2*N
numContext=int((context-1)/2)
pathdir=pathdir_evalita+'AUDIO_FILES/'
micArray,num,fi,lsb,usb=getPhi(nameplace,pathdir,dirArray,dirWall,formattext) 
del lsb,getPhi
fi=np.asarray(fi)
ax=usb[0]*1000.0
ay=usb[1]*1000.0
az=usb[2]*1000.0
ndir=0
numberCV=10