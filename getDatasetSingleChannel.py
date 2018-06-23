from constants import *
import numpy as np
import scipy.io.wavfile as wav
import scipy
from csp2dgpp import getCSP,organizeGCC


def getHSCMADevDataset(VAD_tr_GCC1,SLOC_tr_GCC_X,ax,ay,context,numContext,dirAudio2,fi,nameplace,mic_log_mel):        
    dev_test_array=['Dev']
    dir_dt_array=['HSCMA_DIRHA_dev','HSCMA_DIRHA_test']
    dirrslower=dir_real_sim_array[1]
    for dirdt in dev_test_array:
        print dirdt
        if dirdt=='Dev':
            dirdtupper=dev_test_array[0]
            dirdt_hs=dir_dt_array[0]
        else:
            dirdtupper=dev_test_array[1]
            dirdt_hs=dir_dt_array[1] 
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
                for i in range(l):
                    if POSSOURCE[i][0]!=0 or  POSSOURCE[i][1]!=0 or POSSOURCE[i][2]!=0:
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
                        SLOC_tr_GCC_X.append([float(POSSOURCE[i][0])/ax,float(POSSOURCE[i][1])/ay])



def getEVALITADataset(VAD_tr_GCC1,SLOC_tr_GCC_X,ax,ay,realcheck,context,numContext,dirAudio2,fi,nameplace,mic_log_mel):        
    for realorsim in dir_real_sim_array:
        if (realorsim=='real' and realcheck==True) or realorsim=='sim':
            print realorsim
            dirrslower=realorsim
            if realorsim=='real':
                dirrs=real_sim_array[0]
                numberdir=22
                numbertraining=numberdir
                numberdirtraining=np.linspace(1,numberdir,numberdir)
            elif realorsim=='sim':
                dirrs=real_sim_array[1]
                numberdir=80
                numberdirtraining=np.linspace(1,numberdir,numberdir)
            for numerocartelle in numberdirtraining:
            #for numerocartelle in range(1,11):
                print 'training folder number: '+str(numerocartelle)   
                dira=pathdir+dirrs+'/'+dirrslower+str(int(numerocartelle))+dirAudio2
                from funzioniSupplementari import getMic
                AUDIO=getMic(nameplace,dira)
                del getMic
                AUDIO16=[] 
                for i in range(0,mic_log_mel):
                    AUDIO16.append(scipy.signal.decimate(AUDIO[i],48/16))                 
                l=len(AUDIO16[0])/hopframe
                if realorsim=='sim' or (realorsim=='real' and (nameplace=='Livingroom' or nameplace=='Kitchen')):
                    filesource=pathdir+dirrs+'/'+dirrslower+str(int(numerocartelle))+'/'+dirSource2+'/'+nameplace+formatref
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
                for i in range(l):
                    if POSSOURCE[i][0]!=0 or  POSSOURCE[i][1]!=0 or POSSOURCE[i][2]!=0:
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
                        SLOC_tr_GCC_X.append([float(POSSOURCE[i][0])/ax,float(POSSOURCE[i][1])/ay])                        

def getDLSDataset(VAD_tr_GCC1,SLOC_tr_GCC_X,ax,ay,numberdir_fittizio,pathdir_fittizio,context,numContext,dirAudio2,fi,nameplace,mic_log_mel):        
    for numerocartelle in range(1,numberdir_fittizio):
        print 'training folder number: '+str(numerocartelle)   
        dira=pathdir_fittizio+'audio'+str(int(numerocartelle))+'/'+'audio'+str(int(numerocartelle)) 
        from funzioniSupplementari import getMic16
        AUDIO16=getMic16(nameplace,dira)
        #print AUDIO16
        del getMic16
        l=len(AUDIO16[0])/hopframe
        AUDIO16=np.asarray(AUDIO16)
        filesource=pathdir_fittizio+'audio'+str(int(numerocartelle))+'/'+'audio'+str(int(numerocartelle))+formattext
        from funzioniSupplementari import positionSource16
        POSSOURCE=positionSource16(filesource)
        del positionSource16,filesource
        CSPREAL=[]
        POSTEMP=[]
        for i in range(l):
            POSTEMP.append(POSSOURCE[0])
        POSSOURCE=np.asarray(POSTEMP)
        del POSTEMP
        getCSP(l,lun,N,fi,AUDIO16,window,CSPREAL)
        CSPREAL=np.asarray(CSPREAL)
        CSPTEMP=organizeGCC(CSPREAL,fi,kmax,l)
        del CSPREAL,AUDIO16
        for i in range(l):
            if POSSOURCE[i][0]!=0 or  POSSOURCE[i][1]!=0 or POSSOURCE[i][2]!=0:
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
                SLOC_tr_GCC_X.append([float(POSSOURCE[i][0])/ax,float(POSSOURCE[i][1])/ay])  
