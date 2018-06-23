import os.path
import sys
import numpy as np
def startSimulationSingleChannel(context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,nameplace,dir_training,ax,ay,numContext,dirAudio2,fi,mic_log_mel,hsc,fittizio,simulatedcheck,realcheck,numberdir_fittizio,pathdir_fittizio):
    
    pathsave='Single_Channel'+dir_training+'/'
    pathsave=pathsave+'context'+str(context)+'/'
    pathsave=pathsave+nameplace+'/'
    pathsave=pathsave+'CNNkernel'+str(int(CNNkernel[0]))+'/'
    pathsave=pathsave+'sizeKernel'+str(int(sizekernelCNN))+'/'
    pathsave=pathsave+'strides'+str(int(stridesCNN))+'/'
    pathsave=pathsave+'Densenumber'+str(int(len(DenseNeuron)))+'/'
    pathsave=pathsave+'Denseneuron'+str(int(DenseNeuron[0]))+'/'

    if not os.path.exists(pathsave):
        os.makedirs(pathsave)
    VAD_tr_GCC1=[]
    SLOC_tr_GCC_X=[]
    from getDatasetSingleChannel import getHSCMADevDataset
    if hsc:
        from getDatasetSingleChannel import getHSCMADevDataset
        getHSCMADevDataset(VAD_tr_GCC1,SLOC_tr_GCC_X,ax,ay,context,numContext,dirAudio2,fi,nameplace,mic_log_mel)
    if simulatedcheck:
        from getDatasetSingleChannel import getEVALITADataset
        getEVALITADataset(VAD_tr_GCC1,SLOC_tr_GCC_X,ax,ay,realcheck,context,numContext,dirAudio2,fi,nameplace,mic_log_mel)
    if fittizio:
        from getDatasetSingleChannel import getDLSDataset
        getDLSDataset(VAD_tr_GCC1,SLOC_tr_GCC_X,ax,ay,numberdir_fittizio,pathdir_fittizio,context,numContext,dirAudio2,fi,nameplace,mic_log_mel)     

    VAD_tr_GCC1=np.asarray(VAD_tr_GCC1)
    SLOC_tr_GCC_X=np.asarray(SLOC_tr_GCC_X) 

    indexes=np.random.permutation(VAD_tr_GCC1.shape[0])
    VAD_tr_GCC=VAD_tr_GCC1[indexes]
    print VAD_tr_GCC.shape
    SLOC_tr_GCC=SLOC_tr_GCC_X[indexes]
    del indexes,SLOC_tr_GCC_X,VAD_tr_GCC1
    from getModel import getModelSingleChannel
    model=getModelSingleChannel(CNNkernel,sizekernelCNN,DenseNeuron,stridesCNN,fi,context,VAD_tr_GCC,SLOC_tr_GCC,pathsave,nameplace)
    
    
    
    from constants import *
    import scipy.io.wavfile as wav
    import scipy
    from csp2dgpp import getCSP,organizeGCC
    import cPickle

    dev_test_array=['Dev','Test']
    dir_dt_array=['HSCMA_DIRHA_dev','HSCMA_DIRHA_test']
    dir_real_sim_array=['real','sim']
    dirrslower=dir_real_sim_array[1]
    for dirdt in dev_test_array:
        print dirdt
        if dirdt=='Dev':
            dirdtupper=dev_test_array[0]
            dirdt_hs=dir_dt_array[0]
            devortest='dev'
        else:
            dirdtupper=dev_test_array[1]
            dirdt_hs=dir_dt_array[1]
            devortest='test'
        realorsim=dir_real_sim_array[1]
        dirrs=real_sim_array[1]
        lan_array=['IT','GK','GE','PT']
        numberdir_hscma=10
        numberdirtraining_hscma=np.linspace(1,numberdir_hscma,numberdir_hscma)
        for lan in lan_array: 
            print lan 
            dirAudio=pathdir_hscma+dirdt_hs+'/'+dirrs+'/'+dirdtupper+'/'+lan+'/'
            for numerocartelle in numberdirtraining_hscma:
                print 'training number: '+str(int(numerocartelle))
                dira=dirAudio+dirrslower+str(int(numerocartelle))+dirAudio2  
                from funzioniSupplementari import getMic
                AUDIO=getMic(nameplace,dira)
                del getMic
                AUDIO16=[] 
                for i in range(0,mic_log_mel):
                    AUDIO16.append(scipy.signal.decimate(AUDIO[i],48/16))                 
                l=len(AUDIO16[0])/hopframe
                if realorsim=='sim' or (realorsim=='real' and (nameplace=='Livingroom' or nameplace=='Kitchen')):
                    filesource=dirAudio+dirrslower+str(int(numerocartelle))+'/'+dirSource2+'/'+nameplace+formatref
                    from funzioniSupplementari import positionSource
                    POSSOURCE=positionSource(filesource)
                    del positionSource
                CSPREAL=[]
                POSTEMP=[]
                for i in range(len(POSSOURCE)):
                    POSTEMP.append(POSSOURCE[i])
                POSSOURCE=np.asarray(POSTEMP)
                del POSTEMP
                getCSP(l,lun,N,fi,AUDIO16,window,CSPREAL)
                CSPREAL=np.asarray(CSPREAL)
                CSPTEMP=organizeGCC(CSPREAL,fi,kmax,l)
                del CSPREAL
                VAD_tr_GCC1=[]
                for i in range(l):
                    MatrixInput1=np.zeros(shape=(len(fi),context,kmax))                     
                    n=0
                    while n<context:
                        t=0
                        if (i-numContext+n<0) or (i-numContext+n)>=l:
                            t=i
                        else:
                            t=i-numContext+n
                        for j in range(len(fi)):
                            MatrixInput1[j,n]=CSPTEMP[t,j]
                        n=n+1
                    VAD_tr_GCC1.append(MatrixInput1)
                VAD_tr_GCC1=np.asarray(VAD_tr_GCC1)
                SLOC_testx=model.predict([VAD_tr_GCC1])     
                POSSLOC=[]
                for i in range(len(SLOC_testx)):
                    Px=SLOC_testx[i][0]*float(ax)
                    Py=SLOC_testx[i][1]*float(ay)
                    POSSLOC.append([Px,Py,1500])
                POSSLOC=np.asarray(POSSLOC)
                s=pathsave+realorsim+'_'+devortest+'_'+lan+'_'+str(int(numerocartelle))+'_'+nameplace+'_'+'oracle'+'_'+'singlechannel'
                locfile=s+'.loc'
                loc=open(locfile,'w')
                cPickle.dump(POSSLOC,loc)
                loc.close()                             
    del model



def startSimulationMultiChannel(context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,nameplace,dir_training,ax,ay,numContext,dirAudio2,fi,mic_log_mel,hsc,fittizio,simulatedcheck,realcheck,numberdir_fittizio,pathdir_fittizio):
    
    pathsave='Multi_Channel'+dir_training+'/'
    pathsave=pathsave+'context'+str(context)+'/'
    pathsave=pathsave+nameplace+'/'
    pathsave=pathsave+'CNNkernel'+str(int(CNNkernel[0]))+'/'
    pathsave=pathsave+'sizeKernel'+str(int(sizekernelCNN))+'/'
    pathsave=pathsave+'strides'+str(int(stridesCNN))+'/'
    pathsave=pathsave+'Densenumber'+str(int(len(DenseNeuron)))+'/'
    pathsave=pathsave+'Denseneuron'+str(int(DenseNeuron[0]))+'/'

    if not os.path.exists(pathsave):
        os.makedirs(pathsave)
    
    SLOC_tr_GCC_X=[]
    VAD_tr_GCC1=[]
    for k in range(len(fi)):
        VAD_tr_GCC1.append([])
    from getDatasetMultiChannel import getHSCMADevDataset
    if hsc:
        from getDatasetMultiChannel import getHSCMADevDataset
        getHSCMADevDataset(VAD_tr_GCC1,SLOC_tr_GCC_X,ax,ay,context,numContext,dirAudio2,fi,nameplace,mic_log_mel)
    if simulatedcheck:
        from getDatasetMultiChannel import getEVALITADataset
        getEVALITADataset(VAD_tr_GCC1,SLOC_tr_GCC_X,ax,ay,realcheck,context,numContext,dirAudio2,fi,nameplace,mic_log_mel)
    if fittizio:
        from getDatasetMultiChannel import getDLSDataset
        getDLSDataset(VAD_tr_GCC1,SLOC_tr_GCC_X,ax,ay,numberdir_fittizio,pathdir_fittizio,context,numContext,dirAudio2,fi,nameplace,mic_log_mel)     
 
    VAD_tr_GCC1=np.asarray(VAD_tr_GCC1)
    SLOC_tr_GCC_X=np.asarray(SLOC_tr_GCC_X) 
    indexes=np.random.permutation(VAD_tr_GCC1.shape[1])
    for i in range(len(VAD_tr_GCC1)):
        VAD_tr_GCC1[i]=VAD_tr_GCC1[i][indexes]
    VAD_tr_GCC=[]
    for i in range(len(fi)):
        VAD_tr_GCC.append(VAD_tr_GCC1[i])
    SLOC_tr_GCC=SLOC_tr_GCC_X[indexes]
    del indexes,SLOC_tr_GCC_X,VAD_tr_GCC1
    from getModel import getModelMultiChannel
    model=getModelMultiChannel(CNNkernel,sizekernelCNN,DenseNeuron,stridesCNN,fi,context,VAD_tr_GCC,SLOC_tr_GCC,pathsave,nameplace)
    
    
    
    from constants import *
    import scipy.io.wavfile as wav
    import scipy
    from csp2dgpp import getCSP,organizeGCC
    import cPickle

    dev_test_array=['Dev','Test']
    dir_dt_array=['HSCMA_DIRHA_dev','HSCMA_DIRHA_test']
    dirrslower=dir_real_sim_array[1]
    for dirdt in dev_test_array:
        print dirdt
        if dirdt=='Dev':
            dirdtupper=dev_test_array[0]
            dirdt_hs=dir_dt_array[0]
            devortest='dev'
        else:
            dirdtupper=dev_test_array[1]
            dirdt_hs=dir_dt_array[1]
            devortest='test'
        realorsim=dir_real_sim_array[1]
        dirrs=real_sim_array[1]
        lan_array=['IT','GK','GE','PT']
        numberdir_hscma=10
        numberdirtraining_hscma=np.linspace(1,numberdir_hscma,numberdir_hscma)
        for lan in lan_array: 
            print lan 
            dirAudio=pathdir_hscma+dirdt_hs+'/'+dirrs+'/'+dirdtupper+'/'+lan+'/'
            for numerocartelle in numberdirtraining_hscma:
                print 'training number: '+str(int(numerocartelle))
                dira=dirAudio+dirrslower+str(int(numerocartelle))+dirAudio2  
                from funzioniSupplementari import getMic
                AUDIO=getMic(nameplace,dira)
                del getMic
                AUDIO16=[] 
                for i in range(0,mic_log_mel):
                    AUDIO16.append(scipy.signal.decimate(AUDIO[i],48/16))                 
                l=len(AUDIO16[0])/hopframe
                if realorsim=='sim' or (realorsim=='real' and (nameplace=='Livingroom' or nameplace=='Kitchen')):
                    filesource=dirAudio+dirrslower+str(int(numerocartelle))+'/'+dirSource2+'/'+nameplace+formatref
                    from funzioniSupplementari import positionSource
                    POSSOURCE=positionSource(filesource)
                    del positionSource
                CSPREAL=[]
                POSTEMP=[]
                for i in range(len(POSSOURCE)):
                    POSTEMP.append(POSSOURCE[i])
                POSSOURCE=np.asarray(POSTEMP)
                del POSTEMP
                getCSP(l,lun,N,fi,AUDIO16,window,CSPREAL)
                CSPREAL=np.asarray(CSPREAL)
                CSPTEMP=organizeGCC(CSPREAL,fi,kmax,l)
                VAD_tr_GCC1=[]
                for i in range(len(fi)):
                    VAD_tr_GCC1.append([])
                del CSPREAL
                for i in range(l):
                    MatrixInput1=np.zeros(shape=(len(fi),context,kmax))                     
                    n=0
                    while n<context:
                        t=0
                        if (i-numContext+n<0) or (i-numContext+n)>=l:
                            t=i
                        else:
                            t=i-numContext+n
                        for j in range(len(fi)):
                            MatrixInput1[j,n]=CSPTEMP[t,j]
                        n=n+1
                    for j in range(len(fi)):
                        VAD_tr_GCC1[j].append(MatrixInput1[j])
                VAD_tr_GCC1=np.asarray(VAD_tr_GCC1)
                VAD_tr_GCC=[]
                for i in range(len(fi)):
                    VAD_tr_GCC.append(VAD_tr_GCC1[i])    
                SLOC_testx=model.predict(VAD_tr_GCC)     
                POSSLOC=[]
                for i in range(len(SLOC_testx)):
                    Px=SLOC_testx[i][0]*float(ax)
                    Py=SLOC_testx[i][1]*float(ay)
                    POSSLOC.append([Px,Py,1500])
                POSSLOC=np.asarray(POSSLOC)
                s=pathsave+realorsim+'_'+devortest+'_'+lan+'_'+str(int(numerocartelle))+'_'+nameplace+'_'+'oracle'+'_'+'multichannel'
                locfile=s+'.loc'
                loc=open(locfile,'w')
                cPickle.dump(POSSLOC,loc)
                loc.close()                                      
    del model


def startSimulationJoint(context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer,nameplace,dir_training,ax,ay,numContext,dirAudio2,fi,mic_log_mel,hsc,fittizio,simulatedcheck,realcheck,numberdir_fittizio,pathdir_fittizio):
    
    pathsave='Joint'+dir_training+'/'
    pathsave=pathsave+'context'+str(context)+'/'
    pathsave=pathsave+nameplace+'/'
    pathsave=pathsave+'CNNkernel'+str(int(CNNkernel[0]))+'/'
    pathsave=pathsave+'sizeKernel'+str(int(sizekernelCNN))+'/'
    pathsave=pathsave+'strides'+str(int(stridesCNN))+'/'
    pathsave=pathsave+'Densenumber'+str(int(len(DenseNeuron)))+'/'
    pathsave=pathsave+'Denseneuron'+str(int(DenseNeuron[0]))+'/'
    pathsave=pathsave+'Dropout'+str(int(len(DropoutLayer)))+'/'

    if not os.path.exists(pathsave):
        os.makedirs(pathsave)
    VAD_tr_GCC1=[]
    SLOC_tr_GCC_X=[]
    VAD_tr_LOG1=[]
    SLOC_tr_LOG_X=[]
    from getDatasetJoint import getHSCMADevDataset
    if hsc:
        from getDatasetJoint import getHSCMADevDataset
        getHSCMADevDataset(VAD_tr_LOG1,VAD_tr_GCC1,SLOC_tr_LOG_X,SLOC_tr_GCC_X,ax,ay,context,numContext,dirAudio2,fi,nameplace,mic_log_mel)
    if simulatedcheck:
        from getDatasetJoint import getEVALITADataset
        getEVALITADataset(VAD_tr_LOG1,VAD_tr_GCC1,SLOC_tr_LOG_X,SLOC_tr_GCC_X,ax,ay,realcheck,context,numContext,dirAudio2,fi,nameplace,mic_log_mel)
    if fittizio:
        from getDatasetJoint import getDLSDataset
        getDLSDataset(VAD_tr_LOG1,VAD_tr_GCC1,SLOC_tr_LOG_X,SLOC_tr_GCC_X,ax,ay,numberdir_fittizio,pathdir_fittizio,context,numContext,dirAudio2,fi,nameplace,mic_log_mel)     
    VAD_tr_GCC1=np.asarray(VAD_tr_GCC1)
    SLOC_tr_GCC_X=np.asarray(SLOC_tr_GCC_X)
    VAD_tr_LOG1=np.asarray(VAD_tr_LOG1)
    SLOC_tr_LOG_X=np.asarray(SLOC_tr_LOG_X)
    indexes=np.random.permutation(VAD_tr_GCC1.shape[0])
    VAD_tr_GCC=VAD_tr_GCC1[indexes]
    del VAD_tr_GCC1
    print VAD_tr_GCC.shape
    SLOC_tr_GCC=SLOC_tr_GCC_X[indexes]
    del SLOC_tr_GCC_X
    VAD_tr_LOG=VAD_tr_LOG1[indexes]
    del VAD_tr_LOG1
    SLOC_tr_LOG=SLOC_tr_LOG_X[indexes]
    del SLOC_tr_LOG_X, indexes

    from getModel import getModelJoint
    model=getModelJoint(CNNkernel,sizekernelCNN,DenseNeuron,DropoutLayer,stridesCNN,fi,context,mic_log_mel,VAD_tr_LOG,VAD_tr_GCC,SLOC_tr_LOG,SLOC_tr_GCC,pathsave,nameplace)
    
    
    
    from constants import *
    import scipy.io.wavfile as wav
    import scipy
    from csp2dgpp import getCSP,organizeGCC
    import cPickle
    from python_speech_features import logfbank
    import sklearn
    from sklearn import mixture
    dev_test_array=['Dev','Test']
    dir_dt_array=['HSCMA_DIRHA_dev','HSCMA_DIRHA_test']
    dirrslower=dir_real_sim_array[1]
    for dirdt in dev_test_array:
        print dirdt
        if dirdt=='Dev':
            dirdtupper=dev_test_array[0]
            dirdt_hs=dir_dt_array[0]
            devortest='dev'
        else:
            dirdtupper=dev_test_array[1]
            dirdt_hs=dir_dt_array[1]
            devortest='test'
        realorsim=dir_real_sim_array[1]
        dirrs=real_sim_array[1]
        lan_array=['IT','GK','GE','PT']
        numberdir_hscma=10
        numberdirtraining_hscma=np.linspace(1,numberdir_hscma,numberdir_hscma)
        for lan in lan_array: 
            print lan 
            dirAudio=pathdir_hscma+dirdt_hs+'/'+dirrs+'/'+dirdtupper+'/'+lan+'/'
            for numerocartelle in numberdirtraining_hscma:
                print 'testing number: '+str(int(numerocartelle))
                dira=dirAudio+dirrslower+str(int(numerocartelle))+dirAudio2  
                from funzioniSupplementari import getMic
                AUDIO=getMic(nameplace,dira)
                del getMic
                AUDIO16=[] 
                for i in range(0,mic_log_mel):
                    AUDIO16.append(scipy.signal.decimate(AUDIO[i],48/16))                 
                l=len(AUDIO16[0])/hopframe
                if realorsim=='sim' or (realorsim=='real' and (nameplace=='Livingroom' or nameplace=='Kitchen')):
                    filesource=dirAudio+dirrslower+str(int(numerocartelle))+'/'+dirSource2+'/'+nameplace+formatref
                    from funzioniSupplementari import positionSource
                    POSSOURCE=positionSource(filesource)
                    del positionSource
                CSPREAL=[]
                POSTEMP=[]
                for i in range(len(POSSOURCE)):
                    POSTEMP.append(POSSOURCE[i])
                POSSOURCE=np.asarray(POSTEMP)
                del POSTEMP
                getCSP(l,lun,N,fi,AUDIO16,window,CSPREAL)
                CSPREAL=np.asarray(CSPREAL)
                CSPTEMP=organizeGCC(CSPREAL,fi,kmax,l)
                del CSPREAL
                LOGMEL=[]
                for i in range(0,mic_log_mel):
                    vector=logfbank(AUDIO16[i],rate,nfilt=numfilter,winlen=(float(lenframe)/float(rate)),winstep=(float(hopframe)/float(rate)),nfft=lenframe)
                    vector=sklearn.preprocessing.normalize(vector)                         
                    LOGMEL.append(vector) 
                VAD_tr_LOG1=[]
                VAD_tr_GCC1=[]
                for i in range(l):
                    MatrixInputLOG=np.zeros(shape=(mic_log_mel,context,numfilter))  
                    MatrixInputGCC=np.zeros(shape=(len(fi),context,kmax))                     
                    n=0
                    while n<context:
                        t=0
                        if (i-numContext+n<0) or (i-numContext+n)>=l:
                            t=i
                        else:
                            t=i-numContext+n
                        for j in range(mic_log_mel):
                            MatrixInputLOG[j,n]=LOGMEL[j][t]
                        for j in range(len(fi)):
                            MatrixInputGCC[j,n]=CSPTEMP[t,j]
                        n=n+1
                    VAD_tr_LOG1.append(MatrixInputLOG)
                    VAD_tr_GCC1.append(MatrixInputGCC)    
                VAD_tr_GCC1=np.asarray(VAD_tr_GCC1)
                VAD_tr_LOG1=np.asarray(VAD_tr_LOG1)
                [VAD_output,SLOC_OUTPUT]=model.predict([VAD_tr_LOG1,VAD_tr_GCC1])     
                POSSLOC=[]
                for i in range(len(SLOC_OUTPUT)):
                    Px=SLOC_OUTPUT[i][0]
                    Py=SLOC_OUTPUT[i][1]
                    POSSLOC.append([Px,Py,1500])
                POSSLOC=np.asarray(POSSLOC)
                VAD_output=np.asarray(VAD_output)
                VAD_output=VAD_output[:,1]
                s=pathsave+realorsim+'_'+devortest+'_'+lan+'_'+str(int(numerocartelle))+'_'+nameplace+'_'+'oracle'+'_'+'slocjoint'
                locfile=s+'.loc'
                loc=open(locfile,'w')
                cPickle.dump(POSSLOC,loc)
                loc.close()      
                s=pathsave+realorsim+'_'+devortest+'_'+lan+'_'+str(int(numerocartelle))+'_'+nameplace+'_'+'vadjoint'                   
                vadfile=s+'.spk'
                vad=open(vadfile,'w')
                cPickle.dump(VAD_output,vad)
                vad.close()                                   
    del model