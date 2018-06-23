from funzioniSupplementari import positionSource,audioSource,positionSys
import cPickle
import os.path
import scipy.io.wavfile as wav
import scipy
from scipy import signal
import numpy
from sklearn import mixture
import numpy as np
from scipy.signal import lfilter

def getrealspeech(POSSOURCE):
    l=len(POSSOURCE)
    checkspeech=np.zeros(l)
    for i in range(l):
        if POSSOURCE[i][0]!=0 or POSSOURCE[i][1]!=0 or POSSOURCE[i][2]!=0:
            checkspeech[i]=1
    return checkspeech

def smooth_one_vector_IIR(y,n):
    b=[1.0/n]*n
    a=1
    y_smooth=lfilter(b,a,y)
    return y_smooth

def smooth_one_vector(y,box_pts):
    box=np.ones(box_pts)/box_pts
    y_smooth=np.convolve(y,box,mode='same')
    return y_smooth

def getVad(thres,pathsave,dirdt,lan,POSSOURCE,realorsim,numerocartelle,nameplace,vad='oracle'):
    filejointstring=pathsave+realorsim+'_'+str(numerocartelle)+'_'+'dir_'+dirdt+'_lan_'+lan+'_'+nameplace+'_'+vad+'.spk'    
    checkvad=cPickle.load(open(filejointstring,'r'))
    pr=thres
    vs=checkvad
    vs2=[]
    for i in range(0,len(vs)):
        if vs[i]>pr:
            vs2.append(int(1))
        else:
            vs2.append(int(0))
    checkvad=vs2
    checkvad=np.asarray(checkvad)
    return checkvad

def getVadOracle(POSSOURCE,realorsim,numerocartelle,nameplace,vad='oracle'):
    checkvad=[]
    for i in range(0,len(POSSOURCE)):
        if POSSOURCE[i][0]!=0 or  POSSOURCE[i][1]!=0 or POSSOURCE[i][2]!=0:
            checkvad.append(int(1))
        else:
            checkvad.append(int(0))
    checkvad=np.asarray(checkvad)
    return checkvad



def analizeSLOCVAD(checkspeech,checkvad,POSSOURCE,loc):
    l=len(POSSOURCE)
    checks=checkspeech*checkvad
    tp=0
    tn=0
    fp=0
    fn=0
    fe=[]
    ge=[]
    counter=0
    total=0
    nspeech=0
    nnospeech=0
    for i in range(l):
        if checkspeech[i]==1:
            nspeech=nspeech+1
        else:
            nnospeech=nnospeech+1
        if checkspeech[i]==1 and checkvad[i]==1:
            tp=tp+1
        elif checkspeech[i]==0 and checkvad[i]==0:
            tn=tn+1
        elif checkspeech[i]==0 and checkvad[i]==1:
            fp=fp+1
        elif checkspeech[i]==1 and checkvad[i]==0:
            fn=fn+1
    for i in range(l):
        if checks[i]:
            dis=np.sqrt(np.power(POSSOURCE[i][0]-loc[i][0],2)+np.power(POSSOURCE[i][1]-loc[i][1],2))
            print dis
            if dis<=500:
                fe.append(dis)
                counter=counter+1
            else:
                ge.append(dis)
            total=total+1
    fe=np.asarray(fe)
    ge=np.asarray(ge)
    FE=0
    GE=0
    RMS=0
    rmse=0
    if len(fe)!=0:
        FE=calcolaErrori(FE,fe)
    else:
        FE=0
    if len(ge)!=0:
        GE=calcolaErrori(GE,ge)
    else:
        GE=0
    if(len(fe)!=0) or (len(ge)!=0):
        RMS=calcoloRMSE(rmse,fe,ge)
    else:
        RMS=0 
    return tp,tn,fp,fn,FE,GE,RMS,counter,total,nspeech,nnospeech


def analizeTotalSLOCVAD(tparray,tnarray,fparray,fnarray,farray,garray,rmsarray,ris,tot,spck,nospck):
    ft=0
    gt=0
    rms=0
    r=0
    t=0
    tp=0
    tn=0
    fp=0
    fn=0
    Nsp=0
    Nnsp=0
    for i in range(0,len(farray)):
        #print str(spck[i])+' '+str(nospck[i])
        Nsp=Nsp+spck[i]
        Nnsp=Nnsp+nospck[i]
        ft=ft+farray[i]*ris[i]
        gt=gt+garray[i]*(tot[i]-ris[i])
        rms=rms+rmsarray[i]*tot[i]
        r=r+ris[i]
        t=t+tot[i]
        tp=tp+tparray[i]
        tn=tn+tnarray[i]
        fp=fp+fparray[i]
        fn=fn+fnarray[i]
    if r!=0:
        fe=ft/r
    else:
        fe=0
    if( t!=0 or r!=0 )and t!=r:
        ge=gt/(t-r)
    else:
        ge=0
    if t!=0:
        rms=rms/t
        Pcor=float(r)/float(t)
    else:
        rms=0
        Pcor=0
    if tp!=0 or fp!=0:
        Prec=float(tp)/(float(tp)+float(fp))
    else:
        Prec=0
    if tp!=0 or fn!=0:
        Recall=float(tp)/(float(tp)+float(fn))
    else:
        Recall=0
    if Prec!=0 or Recall!=0:
        Fvalue=2*(Prec*Recall)/(Prec+Recall)
    else:
        Fvalue=0
    if fp!=0 or tn!=0:
        FA=float(fp)/float(Nnsp)
    else:
        FA=0
    if Nsp!=0:
        Del=float(fn)/float(Nsp)
        beta=float(Nnsp)/float(Nsp)
    else:
        Del=0
        beta=0
    SAD=float(fp+beta*fn)/float(Nnsp+beta*Nsp)
    return Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD
def calcoloRMSE(rmse,fineerror,grosserror):
    RMS=(np.sum(fineerror)+np.sum(grosserror))/(len(fineerror)+len(grosserror))
    for i in range(0,len(fineerror)):
        rmse=rmse+np.power((fineerror[i]),2)
    for i in range(0,len(grosserror)):
        rmse=rmse+np.power((grosserror[i]),2)
    rmse=np.sqrt(rmse/(len(fineerror)+len(grosserror)))
    return rmse

def calcolaErrori(fermse,fineerror):
    for i in range(0,len(fineerror)):
        fermse=fermse+np.power((fineerror[i]),2)
    fermse=np.sqrt(fermse/(float(len(fineerror))))
    return fermse


from impostazioni import *

def conclusionSLOCOracle(pathsave,nameplace,devortest,realorsim,method):
    print realorsim
    print nameplace

    if devortest=='dev':
        dirdt='HSCMA_DIRHA_dev'
        dirdtupper='Dev'
    elif devortest=='test':
        dirdt='HSCMA_DIRHA_test'
        dirdtupper='Test'
    if devortest=='dev' and realorsim=='real':
        numberdir=12
    elif (devortest=='dev' or devortest=='test') and realorsim=='sim':
        numberdir=10
    elif devortest=='test' and realorsim=='real':
        numberdir=10
    if realorsim=='real':
        language=['IT']
        dirrs='Real'
        dirrslower='real'
    else:
        language=['IT','PT','GE','GK']
        dirrs='Simulations'
        dirrslower='sim'
        tparray=[]
        tnarray=[]
        fparray=[]
        fnarray=[]
        fearray=[]
        gearray=[]
        rmsarray=[]
        counterarray=[]
        totalarray=[]   
        speecharray=[]
        speechnoarray=[]    
    for lan in language:
        for numerocartelle in range(1,numberdir+1):
            dirAudio=pathdir_hscma+dirdt+'/'+dirrs+'/'+dirdtupper+'/'+lan+'/' 
            locfile=pathsave+realorsim+'_'+devortest+'_'+lan+'_'+str(numerocartelle)+'_'+nameplace+'_'+'oracle'+'_'+method+'.loc'
            loc=cPickle.load(open(locfile,'r'))
            filesource=dirAudio+dirrslower+str(numerocartelle)+'/'+dirSource2+'/'+nameplace+formatref  
            POSSOURCE=positionSource(filesource)
            speech=getrealspeech(POSSOURCE)
            filtrare=False
            if filtrare==True:
                xl=np.zeros(len(loc))
                yl=np.zeros(len(loc))
                for i in range(len(loc)):
                    xl[i]=loc[i][0]
                    yl[i]=loc[i][1]
                xl=smooth_one_vector_IIR(xl,5)
                yl=smooth_one_vector_IIR(yl,5)
                loc[:,0]=xl
                loc[:,1]=yl

            checkvad=getVadOracle(POSSOURCE,realorsim,numerocartelle,nameplace,'oracle')
            tptemp,tntemp,fptemp,fntemp,FE,GE,RMS,counter,total,speechtemp,nospeechtemp=analizeSLOCVAD(speech,checkvad,POSSOURCE,loc)
            tparray.append(tptemp)
            tnarray.append(tntemp)
            fparray.append(fptemp)
            fnarray.append(fntemp)
            fearray.append(FE)
            gearray.append(GE)
            rmsarray.append(RMS)
            counterarray.append(counter)
            totalarray.append(total)
            speecharray.append(speechtemp)
            speechnoarray.append(nospeechtemp)
                                        
    Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD=analizeTotalSLOCVAD(tparray,tnarray,fparray,fnarray,fearray,gearray,rmsarray,counterarray,totalarray,speecharray,speechnoarray)
    print 'analizing case '+realorsim+ ' method '+ method+' vad ' + 'oracle'+'\n'
    print 'Pcor: %.3f\tFE: %.0f\tGE: %.0f\tRMS: %.0f\tPrecision: %.3f\tRecall: %.3f\tF value: %.3f\ttp: %.0f\ttn: %.0f\tfp: %.0f\tfn: %.0f\tFA: %.3f\tDel: %.3f\tSAD: %.3f\n' %(Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD)
    return Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD

def conclusionSLOCVAD(thres,pathsave,nameplace,devortest,realorsim,method,vad):
    print realorsim
    print nameplace

    if devortest=='dev':
        dirdt='HSCMA_DIRHA_dev'
        dirdtupper='Dev'
    elif devortest=='test':
        dirdt='HSCMA_DIRHA_test'
        dirdtupper='Test'
    if devortest=='dev' and realorsim=='real':
        numberdir=12
    elif (devortest=='dev' or devortest=='test') and realorsim=='sim':
        numberdir=10
    elif devortest=='test' and realorsim=='real':
        numberdir=10
    if realorsim=='real':
        language=['IT']
        dirrs='Real'
        dirrslower='real'
    else:
        language=['IT','PT','GE','GK']
        dirrs='Simulations'
        dirrslower='sim'
        tparray=[]
        tnarray=[]
        fparray=[]
        fnarray=[]
        fearray=[]
        gearray=[]
        rmsarray=[]
        counterarray=[]
        totalarray=[]   
        speecharray=[]
        speechnoarray=[]    
    for lan in language:
        for numerocartelle in range(1,numberdir+1):
            dirAudio=pathdir_hscma+dirdt+'/'+dirrs+'/'+dirdtupper+'/'+lan+'/' 
            locfile=pathsave+realorsim+'_'+devortest+'_'+lan+'_'+str(numerocartelle)+'_'+nameplace+'_'+'oracle'+'_'+method+'.loc'
            loc=cPickle.load(open(locfile,'r'))
            filesource=dirAudio+dirrslower+str(numerocartelle)+'/'+dirSource2+'/'+nameplace+formatref  
            POSSOURCE=positionSource(filesource)
            speech=getrealspeech(POSSOURCE)
            filtrare=False
            if filtrare==True:
                xl=np.zeros(len(loc))
                yl=np.zeros(len(loc))
                for i in range(len(loc)):
                    xl[i]=loc[i][0]
                    yl[i]=loc[i][1]
                xl=smooth_one_vector_IIR(xl,5)
                yl=smooth_one_vector_IIR(yl,5)
                loc[:,0]=xl
                loc[:,1]=yl

            checkvad=getVad(thres,pathsave,dirdt,lan,POSSOURCE,realorsim,numerocartelle,nameplace,vad)
            tptemp,tntemp,fptemp,fntemp,FE,GE,RMS,counter,total,speechtemp,nospeechtemp=analizeSLOCVAD(speech,checkvad,POSSOURCE,loc)
            tparray.append(tptemp)
            tnarray.append(tntemp)
            fparray.append(fptemp)
            fnarray.append(fntemp)
            fearray.append(FE)
            gearray.append(GE)
            rmsarray.append(RMS)
            counterarray.append(counter)
            totalarray.append(total)
            speecharray.append(speechtemp)
            speechnoarray.append(nospeechtemp)
                                        
    Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD=analizeTotalSLOCVAD(tparray,tnarray,fparray,fnarray,fearray,gearray,rmsarray,counterarray,totalarray,speecharray,speechnoarray)
    print 'analizing case '+realorsim+ ' method '+ method+' vad ' + vad+'\n'
    print 'Pcor: %.3f\tFE: %.0f\tGE: %.0f\tRMS: %.0f\tPrecision: %.3f\tRecall: %.3f\tF value: %.3f\ttp: %.0f\ttn: %.0f\tfp: %.0f\tfn: %.0f\tFA: %.3f\tDel: %.3f\tSAD: %.3f\n' %(Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD)
    return Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD

def evaluate(fileToWrite,pathsave,nameplace,method):
    from constants import *
    micArray,num,fi,lsb,usb=getPhi(nameplace,pathdir,dirArray,dirWall,formattext) 
    ax=usb[0]*1000.0
    ay=usb[1]*1000.0
    az=usb[2]*1000.0
    array_parametersSLOC=np.zeros(14)
    for devortest in ['dev','test']:
        Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD=conclusionSLOCOracle(pathsave,nameplace,devortest,'sim',method)
        array_parametersSLOC=[Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD]
        parameters_string=['Pcor: ','FE: ','GE: ','RMS: ','Precision: ','Recall: ','F_value: ','tp: ','tn: ','fp: ','fn: ','FA: ','Del: ','SAD: ']
        file = open(pathsave+devortest+'oracleresult.txt','w') 
        fileresults=open(fileToWrite,'a')
        print 'best result with SLOC output: \r'
        file.write('best result with SLOC output: \r')
        fileresults.write(devortest+'\t')
        for i in range(len(parameters_string)):
            print parameters_string[i] +str(array_parametersSLOC[i])+'\r'
            file.write(parameters_string[i] +str(array_parametersSLOC[i])+'\r')
            fileresults.write(parameters_string[i] +str(array_parametersSLOC[i])+'\r')
        fileresults.write('\n')
        file.close()
        fileresults.close()


def evaluationSingleChannel(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron):
    pathmodel='Single_channel/'
    pathsave=pathmodel+'context'+str(context)+'/'
    pathsave=pathsave+nameplace+'/'
    pathsave=pathsave+'CNNkernel'+str(int(CNNkernel[0]))+'/'
    pathsave=pathsave+'sizeKernel'+str(int(sizekernelCNN))+'/'
    pathsave=pathsave+'strides'+str(int(stridesCNN))+'/'
    pathsave=pathsave+'Densenumber'+str(int(len(DenseNeuron)))+'/'
    pathsave=pathsave+'Denseneuron'+str(int(DenseNeuron[0]))+'/'
    stringToWrite='room '+nameplace+'\t'
    stringToWrite=stringToWrite+'context '+str(int(context))+'\t'
    stringToWrite=stringToWrite+'CNN Kernel '
    for i in range(len(CNNkernel)):
        stringToWrite=stringToWrite+str(int(CNNkernel[i]))+'\t'
    stringToWrite=stringToWrite+'size kernel '+str(int(sizekernelCNN))+'\t'
    stringToWrite=stringToWrite+'strides '+str(int(stridesCNN))+'\t'
    stringToWrite=stringToWrite+'Dense '
    for i in range(len(DenseNeuron)):
        stringToWrite=stringToWrite+str(int(DenseNeuron[i]))
    file=open(fileToWrite,'a')
    file.write(stringToWrite+'\n')
    file.close()
    evaluate(fileToWrite,pathsave,nameplace,'singlechannel')

def evaluationMultiChannel(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron):
    pathmodel='Multi_channel/'
    pathsave=pathmodel+'context'+str(context)+'/'
    pathsave=pathsave+nameplace+'/'
    pathsave=pathsave+'CNNkernel'+str(int(CNNkernel[0]))+'/'
    pathsave=pathsave+'sizeKernel'+str(int(sizekernelCNN))+'/'
    pathsave=pathsave+'strides'+str(int(stridesCNN))+'/'
    pathsave=pathsave+'Densenumber'+str(int(len(DenseNeuron)))+'/'
    pathsave=pathsave+'Denseneuron'+str(int(DenseNeuron[0]))+'/'
    stringToWrite='room '+nameplace+'\t'
    stringToWrite=stringToWrite+'context '+str(int(context))+'\t'
    stringToWrite=stringToWrite+'CNN Kernel '
    for i in range(len(CNNkernel)):
        stringToWrite=stringToWrite+str(int(CNNkernel[i]))+'\t'
    stringToWrite=stringToWrite+'size kernel '+str(int(sizekernelCNN))+'\t'
    stringToWrite=stringToWrite+'strides '+str(int(stridesCNN))+'\t'
    stringToWrite=stringToWrite+'Dense '
    for i in range(len(DenseNeuron)):
        stringToWrite=stringToWrite+str(int(DenseNeuron[i]))
    file=open(fileToWrite,'a')
    file.write(stringToWrite+'\n')
    file.close()
    evaluate(fileToWrite,pathsave,nameplace,'multichannel')

def evaluatewithVAD(pathsave,nameplace,method,vad):
    from constants import *
    micArray,num,fi,lsb,usb=getPhi(nameplace,pathdir,dirArray,dirWall,formattext) 
    ax=usb[0]*1000.0
    ay=usb[1]*1000.0
    az=usb[2]*1000.0
    tharray=np.linspace(0,1,int(1.0/0.05))
    SAD_maxVAD=0.5
    array_parametersVAD=np.zeros(15)
    for devortest in ['dev','test']:
        for t in tharray:
            Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD=conclusionSLOCVAD(t,pathsave,nameplace,devortest,'sim',method,vad)
            if SAD_maxVAD>=SAD:
                SAD_maxVAD=SAD
                array_parametersVAD=[t,Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD]
        parameters_string=['threshold: ','Pcor: ','FE: ','GE: ','RMS: ','Precision: ','Recall: ','F_value: ','tp: ','tn: ','fp: ','fn: ','FA: ','Del: ','SAD: ']
        file = open(pathsave+devortest+'result.txt','w') 
        print 'best result with VAD output: \r'
        file.write('best result with VAD output: \r')
        for i in range(len(parameters_string)):
            print parameters_string[i] +str(array_parametersVAD[i])+'\r'
            file.write(parameters_string[i] +str(array_parametersVAD[i])+'\r')
        file.close()