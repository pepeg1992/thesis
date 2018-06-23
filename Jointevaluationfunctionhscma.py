from funzioniSupplementari import positionSource,audioSource,positionSys
import cPickle
import os.path
import scipy.io.wavfile as wav
import scipy
from scipy import signal
import numpy
from sklearn import mixture
import numpy as np
from funzioniSupplementari import getPhi

def getrealspeech(POSSOURCE):
    l=len(POSSOURCE)
    checkspeech=np.zeros(l)
    for i in range(l):
        if POSSOURCE[i][0]!=0 or POSSOURCE[i][1]!=0 or POSSOURCE[i][2]!=0:
            checkspeech[i]=1
    return checkspeech

def getVad(pathsave,thres,POSSOURCE,realorsim,numerocartelle,nameplace,dirdt,lan,vad='oracle',fileIDstring=None):
    filejointstring=pathsave+realorsim+'_'+str(numerocartelle)+'_'+'dir_'+dirdt+'_lan_'+lan+'_'+nameplace+'_'+'vadjoint.spk'    
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
    if not t==0 and not r==0 and not t==r:
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

def VAD_SLOC_threshold_linear(SLOC_output_label, m=-1, q=0):
    # m and q are for    y = m * x + q;
    # reccomended m = -1, q = 0;
    sp  = [0, 1];
    nsp = [1, 0];
    VAD_threshold_output = [];
    for i in range(0,len(SLOC_output_label)):
        x = SLOC_output_label[i,0];
        y = SLOC_output_label[i,1];
        y_compare = x * m + q;
        if y > y_compare:
            VAD_threshold_output.append(sp);
        else:
            VAD_threshold_output.append(nsp);
    return np.asarray(VAD_threshold_output);

def conclusionSLOC(pathsave,nameplace,ax,ay,numberCV,thres,devortest,vad,method,realorsim):
    print realorsim
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
    rmsarray=[]
    counterarray=[]
    totalarray=[]
    gearray=[]
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
            loc=cPickle.load(open(locfile,'r'))
            loctemp=np.copy(loc)
            loctemp[:,0]=(loctemp[:,0]*ax)
            loctemp[:,1]=(loctemp[:,1]*ay) 
            checkvadtemp=VAD_SLOC_threshold_linear(loc, m=-1, q=thres)
            checkvad=checkvadtemp[:,1]
            tptemp,tntemp,fptemp,fntemp,FE,GE,RMS,counter,total,speechtemp,nospeechtemp=analizeSLOCVAD(speech,checkvad,POSSOURCE,loctemp)
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
    print 'analizing case '+realorsim+ ' method '+ method+' vad ' + 'oracle\n'
    print 'Pcor: %.3f\tFE: %.0f\tGE: %.0f\tRMS: %.0f\tPrecision: %.3f\tRecall: %.3f\tF value: %.3f\ttp: %.0f\ttn: %.0f\tfp: %.0f\tfn: %.0f\tFA: %.3f\tDel: %.3f\tSAD: %.3f\n' %(Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD)
    return Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD

def conclusionVAD(pathsave, nameplace,ax,ay,numberCV,thres,devortest,vad,method,realorsim):
    print realorsim
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
    rmsarray=[]
    counterarray=[]
    totalarray=[]
    gearray=[]
    speecharray=[]
    speechnoarray=[]
    for lan in language:
        for numerocartelle in range(1,numberdir+1):
                                #print 'analize number directory: ' +str(numerocartelle)+ ' ...'
            dirAudio=pathdir_hscma+dirdt+'/'+dirrs+'/'+dirdtupper+'/'+lan+'/' 
            locfile=pathsave+realorsim+'_'+devortest+'_'+lan+'_'+str(numerocartelle)+'_'+nameplace+'_'+'oracle'+'_'+method+'.loc'
            loc=cPickle.load(open(locfile,'r'))
            filesource=dirAudio+dirrslower+str(numerocartelle)+'/'+dirSource2+'/'+nameplace+formatref
            POSSOURCE=positionSource(filesource)
            speech=getrealspeech(POSSOURCE) 
            checkvad=getVad(pathsave,thres,POSSOURCE,realorsim,numerocartelle,nameplace,dirdtupper,lan,vad)
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
    print 'analizing case '+devortest+' '+realorsim+ ' method '+ method+' vad ' + vad+'\n'
    print 'Pcor: %.3f\tFE: %.0f\tGE: %.0f\tRMS: %.0f\tPrecision: %.3f\tRecall: %.3f\tF value: %.3f\ttp: %.0f\ttn: %.0f\tfp: %.0f\tfn: %.0f\tFA: %.3f\tDel: %.3f\tSAD: %.3f\n' %(Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD)
    return Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD

from constants import *

def evaluate(fileToWrite,pathsave,nameplace):
    micArray,num,fi,lsb,usb=getPhi(nameplace,pathdir,dirArray,dirWall,formattext) 
    ax=usb[0]*1000.0
    ay=usb[1]*1000.0
    az=usb[2]*1000.0
    tharray=np.linspace(1.0,-1.0,int(2.0/0.02))
    SAD_maxSLOC=0.5
    SAD_maxVAD=0.5
    array_parametersSLOC=np.zeros(15)
    array_parametersVAD=np.zeros(15)
    for devortest in ['dev','test']:
        for t in tharray:
            print t
            Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD=conclusionSLOC(pathsave,nameplace,ax,ay,0,t,devortest,'vadjoint','slocjoint','sim')
            if SAD_maxSLOC>=SAD:
                SAD_maxSLOC=SAD
                array_parametersSLOC=[t,Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD]
            Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD=conclusionVAD(pathsave,nameplace,ax,ay,0,t,devortest,'vadjoint','slocjoint','sim')
            if SAD_maxVAD>=SAD:
                SAD_maxVAD=SAD
                array_parametersVAD=[t,Pcor,fe,ge,rms,Prec,Recall,Fvalue,tp,tn,fp,fn,FA,Del,SAD]
        parameters_string=['threshold: ','Pcor: ','FE: ','GE: ','RMS: ','Precision: ','Recall: ','F_value: ','tp: ','tn: ','fp: ','fn: ','FA: ','Del: ','SAD: ']
        file = open(pathsave+devortest+'result.txt','w') 
        fileresults=open(fileToWrite,'a')
        print 'best result with SLOC output: \r'
        file.write('best result with SLOC output: \r')
        fileresults.write(devortest+'\t')
        for i in range(len(parameters_string)):
            print parameters_string[i] +str(array_parametersSLOC[i])+'\r'
            file.write(parameters_string[i] +str(array_parametersSLOC[i])+'\r')
            fileresults.write(parameters_string[i] +str(array_parametersSLOC[i])+'\r')
        print '\n'
        file.write('\n')
        print 'best result with VAD output: \r'
        file.write('best result with VAD output: \r')
        fileresults.write('\n')
        for i in range(len(parameters_string)):
            print parameters_string[i] +str(array_parametersVAD[i])+'\r'
            file.write(parameters_string[i] +str(array_parametersVAD[i])+'\r')
            fileresults.write(parameters_string[i] +str(array_parametersVAD[i])+'\r')
        file.close()

def evaluationJoint(fileToWrite,nameplace,context,CNNkernel,sizekernelCNN,stridesCNN,DenseNeuron,DropoutLayer):
    pathsave='Joint/'
    pathsave=pathsave+'context'+str(context)+'/'
    pathsave=pathsave+nameplace+'/'
    pathsave=pathsave+'CNNkernel'+str(int(CNNkernel[0]))+'/'
    pathsave=pathsave+'sizeKernel'+str(int(sizekernelCNN))+'/'
    pathsave=pathsave+'strides'+str(int(stridesCNN))+'/'
    pathsave=pathsave+'Densenumber'+str(int(len(DenseNeuron)))+'/'
    pathsave=pathsave+'Denseneuron'+str(int(DenseNeuron[0]))+'/'
    pathsave=pathsave+'DropoutLayer'+str(int(len(DropoutLayer)))
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
    if len(DropoutLayer!=0):
        stringToWrite=stringToWrite+'Dropout '
        for i in range(len(DropoutLayer)):
            stringToWrite=stringToWrite+str(int(DropoutLayer[i]))+'\t'    
    file=open(fileToWrite,'a')
    file.write(stringToWrite+'\n')
    file.close()
    evaluate(fileToWrite,pathsave,nameplace)