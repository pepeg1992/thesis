from funzioniSupplementari import getPhi
import numpy as np
from constants import *
from Jointevaluationfunctionhscma import evaluationJoint
##############      SCEGLIERE IL TIPO DI CONTESTO, DATASET E STANZA     ##############
simulatedcheck=False
realcheck=False
hsc=True
fittizio=False
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
pathdir=pathdir_evalita+'AUDIO_FILES/'
micArray,num,fi,lsb,usb=getPhi(nameplace,pathdir,dirArray,dirWall,formattext) 
del lsb,getPhi
fi=np.asarray(fi)
fileToWrite='Joint'+dir_training+'/risultati.txt'
file=open(fileToWrite,'w')
file.close()
ax=usb[0]*1000.0
ay=usb[1]*1000.0
az=usb[2]*1000.0
context=15
numContext=int((context-1)/2)
CNNkernel=[64]
DenseNeuron=[1024,1024,256]
DropoutLayer=[]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[128]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN)) 
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[64]
DropoutLayer=[0.5]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[128]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
DropoutLayer=[0.5,0.5]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[64]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
DropoutLayer=[0.5,0.5,0.5]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[128]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[64]
DenseNeuron=[1024,1024,1024,256]
DropoutLayer=[]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[128]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN)) 
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[64]
DropoutLayer=[0.5]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[128]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
DropoutLayer=[0.5,0.5]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[64]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
DropoutLayer=[0.5,0.5,0.5]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
CNNkernel=[128]
for sizekernelCNN in [3,4,5]:
    for stridesCNN in [1,2,3,4,5]:
        print 'kernel size: '+str(int(sizekernelCNN))  
        print 'strides : '+str(int(stridesCNN))  
        evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer)
