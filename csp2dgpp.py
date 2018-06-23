from scipy.io import wavfile
from funzioniSupplementari import positionMic, positionSource, audioSource, getCandidate
import numpy as np
import scipy
from scipy import signal
import numpy.fft
import sys
#from numba import jit
#import numba
#from numpy.polynomial import Polynomial
#from scipy.interpolate import UnivariateSpline
#import pyculib.fft
#import cPickle

np.seterr(divide='ignore', invalid='ignore')

#@jit(nopython=True)
def getTDOAtheoric(POSMIC,fi,POSCANDIDATE):
    soundvel=340000.0

    tau=np.zeros(shape=(len(POSCANDIDATE),len(fi)))
    for n in range(0,len(POSCANDIDATE)):
        for i in range(0,len(fi)):
            t=(np.sqrt(np.power(POSMIC[fi[i][0]][0]-POSCANDIDATE[n][0],2)+np.power(POSMIC[fi[i][0]][1]-POSCANDIDATE[n][1],2)+np.power(POSMIC[fi[i][0]][2]-POSCANDIDATE[n][2],2))-np.sqrt(np.power(POSMIC[fi[i][1]][0]-POSCANDIDATE[n][0],2)+np.power(POSMIC[fi[i][1]][1]-POSCANDIDATE[n][1],2)+np.power(POSMIC[fi[i][1]][2]-POSCANDIDATE[n][2],2)))/soundvel
            #t=(np.sqrt(np.power(POSMIC[fi[i][0]][0]-POSCANDIDATE[n][0],2)+np.power(POSMIC[fi[i][0]][1]-POSCANDIDATE[n][1],2))-np.sqrt(np.power(POSMIC[fi[i][1]][0]-POSCANDIDATE[n][0],2)+np.power(POSMIC[fi[i][1]][1]-POSCANDIDATE[n][1],2)))/soundvel

            tau[n,i]=t
    return tau

def csp2dstft(AUDIO16,POSCANDIDATE,POSSOURCE,POSMIC,fi,taustft,checkvad,tautemmod=None,scelta=0):
    l=len(AUDIO16[0])/800
    taucsp = np.zeros(shape=(l,len(fi)))
    N=960
    c=340000.0
    window=np.hanning(N)
    C=scipy.zeros(N)
    #CSP=[]
    #numcsp=5
    #CSPREAL=[]
    CSPTEMP=[]
    lun=(2*N)
    max_shift=int((lun/2))  

    fi=np.asarray(fi)
    POSSOURCE=np.asarray(POSSOURCE)
    POSMIC=np.asarray(POSMIC)    
    getCSP(l,lun,N,fi,AUDIO16,window,CSPTEMP)
    CSPTEMP=np.asarray(CSPTEMP)

    smoothCSP(l,CSPTEMP,fi,max_shift,checkvad)

    taucsp=np.asarray(taucsp)
    getTDOACSP(l,fi,POSSOURCE,POSMIC,CSPTEMP,c,max_shift,taucsp)

    probsource=[]
    P=np.zeros(shape=(l,len(POSCANDIDATE)))
    fineerror=[]
    grosserror=[]
    candidate=0
    total=0
    total,candidate,pot,pos=getCostFunction(l,fi,probsource,P,fineerror,grosserror,candidate,total,POSCANDIDATE,POSSOURCE,taustft,taucsp)
    #del POSSOURCE,POSMIC,CSPTEMP,max_shift,taucsp,fi,P,probsource,   
    fermse=0
    germse=0
    rmse=0
    fineerror=np.asarray(fineerror)
    grosserror=np.asarray(grosserror)
    if(len(fineerror)!=0):
        fermse,FE=calcolaErrori(fermse,fineerror)
    else:
        FE=0
    if(len(grosserror)!=0):
        germse,GE=calcolaErrori(germse,grosserror)
    else:
        GE=0
    if(len(grosserror)!=0) or (len(fineerror)!=0):
        rmse,RMS=calcoloRMSE(rmse,fineerror,grosserror)
    else:
        RMS=0   
    
    if scelta==0:

        return candidate, total,fermse,rmse,pot,pos
    
    elif scelta==1:
        cnterror=np.zeros(shape=(len(POSCANDIDATE),len(fi)))
        upgradeTemplate(l,fi,POSSOURCE,POSCANDIDATE,tautemmod,taucsp,taustft,cnterror)

        return candidate, total,fermse,rmse,pot,pos,cnterror
    elif scelta==4:
        pot=getCostFunctionTraining(l,fi,probsource,P,fineerror,grosserror,candidate,total,POSCANDIDATE,POSSOURCE,taustft,taucsp)
        return pot
    elif scelta==5:
        cnterror=np.zeros(shape=(len(POSCANDIDATE),len(fi)))
        upgradeTemplate(l,fi,POSSOURCE,POSCANDIDATE,tautemmod,taucsp,taustft,cnterror)
        total,candidate,pot,pos=getCostFunction(l,fi,probsource,P,fineerror,grosserror,candidate,total,POSCANDIDATE,POSSOURCE,taustft,taucsp)
        pot=getCostFunctionTraining(l,fi,probsource,P,fineerror,grosserror,candidate,total,POSCANDIDATE,POSSOURCE,taustft,taucsp)
        return pot


##@jit(nopython=True)
def upgradeTemplate(l,fi,POSSOURCE,POSCANDIDATE,tautemmod,taucsp,taustft,cnterror):
    for i in range(0,l):
        if(POSSOURCE[i][0]!=0) and (POSSOURCE[i][1]!=0) and (POSSOURCE[i][2]!=0):  
            for n in range(0,len(POSCANDIDATE)):
                disx=np.sqrt(np.power(POSSOURCE[i][0]-POSCANDIDATE[n][0],2))
                disy=np.sqrt(np.power(POSSOURCE[i][1]-POSCANDIDATE[n][1],2))
                disz=np.sqrt(np.power(POSSOURCE[i][2]-POSCANDIDATE[n][2],2))   
                if disx<125 and disy<125 and disz<125:
                    for j in range(0,len(fi)):     
                        tautemmod[n][j]=tautemmod[n][j]+taucsp[i][j]-taustft[n][j]
                        cnterror[n][j]=cnterror[n][j]+1



#@#jit(nopython=True)
def calcolaErrori(fermse,fineerror):
    for i in range(0,len(fineerror)):
        fermse=fermse+np.power((fineerror[i]),2)
    fermse=np.sqrt(fermse/(float(len(fineerror))))
    FE=fermse
    return fermse,FE


#@jit(nopython=True)
def calcoloRMSE(rmse,fineerror,grosserror):
    RMS=(np.sum(fineerror)+np.sum(grosserror))/(len(fineerror)+len(grosserror))
    for i in range(0,len(fineerror)):
        rmse=rmse+np.power((fineerror[i]),2)
    for i in range(0,len(grosserror)):
        rmse=rmse+np.power((grosserror[i]),2)
    rmse=np.sqrt(rmse/(len(fineerror)+len(grosserror)))
    RMS=rmse
    return rmse,RMS

def getCSP(l,lun,N,fi,AUDIO16,window,CSPTEMP):
    x=np.arange(0,lun)
    AUDIO16=np.asarray(AUDIO16)
    for k in range(0,l):
        CSPF=[]
        ni=somma(moltiplica(k,800),-(dividi(N,2)))
        nf=somma(moltiplica(k,800),(dividi(N,2)))

        for i in range(0,len(fi)):
            #print fi[i]
            d1,d2=calculateCSP(N,k,i,AUDIO16,fi,ni,nf,l)
            S1=(np.fft.fft(d1*window,n=lun))
            S2=(np.fft.fft(d2*window,n=lun))
            
            NUM=moltiplica(S1,np.conj(S2))
            C=dividi(NUM,np.absolute(NUM)) 
            I=np.fft.irfft(C,n=lun)
            CSPF.append(np.absolute(I))
        CSPTEMP.append(CSPF)

def getCSP10ms(l,lun,N,fi,AUDIO16,window,CSPTEMP):
    AUDIO16=np.asarray(AUDIO16)

    for k in range(0,5*l):
        CSPF=[]
        ni=somma(moltiplica(k,800/5),-(dividi(N,2)))
        nf=somma(moltiplica(k,800/5),(dividi(N,2)))

        for i in range(0,len(fi)):

            d1,d2=calculateCSP10ms(N,k,i,AUDIO16,fi,ni,nf,l)
            S1=(np.fft.fft(d1*window,n=lun))
            S2=(np.fft.fft(d2*window,n=lun))
            
            NUM=moltiplica(S1,np.conj(S2))
            C=dividi(NUM,np.absolute(NUM)) 
            I=np.fft.irfft(C,n=lun)
            CSPF.append(np.absolute(I))
        CSPTEMP.append(CSPF)

#@jit(nopython=True)
def calculateCSP(N,k,i,AUDIO16,fi,ni,nf,l):
    d1=np.zeros(N)
    d2=np.zeros(N)
           
    if (ni<0):
        d1[N/2:]=AUDIO16[fi[i][0]][k*800:k*800+N/2]
        d2[N/2:]=AUDIO16[fi[i][1]][k*800:k*800+N/2]
    elif (nf>l*800):
        d1[0:N/2]=AUDIO16[fi[i][0]][-N/2:]
        d2[0:N/2]=AUDIO16[fi[i][1]][-N/2:]  
    else:
        d1=AUDIO16[fi[i][0]][ni:nf]
        d2=AUDIO16[fi[i][1]][ni:nf]
    return d1,d2

#@jit(nopython=True)
def calculateCSP10ms(N,k,i,AUDIO16,fi,ni,nf,l):
    d1=np.zeros(N)
    d2=np.zeros(N) 
           
    if (ni<0):
        d1[N/2:]=AUDIO16[fi[i][0]][k*800/5:k*800/5+N/2]
        d2[N/2:]=AUDIO16[fi[i][1]][k*800/5:k*800/5+N/2]
    elif (nf>l*800/5):
        d1[0:N/2]=AUDIO16[fi[i][0]][-N/2:]
        d2[0:N/2]=AUDIO16[fi[i][1]][-N/2:]  
    else:
        d1=AUDIO16[fi[i][0]][ni:nf]
        d2=AUDIO16[fi[i][1]][ni:nf]
    return d1,d2

##@jit(nopython=True)
def somma(a,b):
    c=a+b
    return c

##@jit(nopython=True)
def moltiplica(a,b):
    c=a*b
    return c

#@jit(nopython=True)
def dividi(a,b):
    c=a/b
    return c


##@jit(nopython=True)
def smoothCSP(l,CSPTEMP,fi,max_shift,POSSOURCE):
    for k in range(0,l):
        if POSSOURCE[k]==1:
            for i in range(0,len(fi)):
                I=calcolasmoothing(CSPTEMP,k,i,POSSOURCE,l)
                CSPTEMP[k][i]=I

#@jit(nopython=True)
def calcolasmoothing(CSPTEMP,k,i,POSSOURCE,l):
    counter=0
    II=np.zeros(shape=(len(CSPTEMP[k][i])))
    j=k
    fermati=1
    while j>=0 and POSSOURCE[j]==1:
        II=somma(II,CSPTEMP[j][i])
        counter=somma(counter,1)
        j=j-1
    j=k+1
    while POSSOURCE[j]==1 and j<l and fermati:
        II=somma(II,CSPTEMP[j][i])
        counter=somma(counter,1)
        j=j+1
        if j==l:
            fermati=0
            break

    I=dividi(II,counter)
    return I    

#@jit(nopython=True)
def getTDOACSP(l,fi,POSSOURCE,POSMIC,CSPREAL,c,max_shift,taucsp):
    for k in range(0,l):
        for i in range(0,len(fi)):
            d=np.sqrt(np.power(POSMIC[fi[i][0]][0]-POSMIC[fi[i][1]][0],2)+np.power(POSMIC[fi[i][0]][1]-POSMIC[fi[i][1]][1],2)+np.power(POSMIC[fi[i][0]][2]-POSMIC[fi[i][1]][2],2))
            kmax=int((d*(16000.0/c)+1))
            III=(CSPREAL[k][i])
            IIItemp=np.concatenate((III[-kmax:],III[0:kmax+1]))
            n=np.argmax(IIItemp)-kmax
            taucsp[k,i]=n/16000.0

def getCostFunction(l,fi,probsource,P,fineerror,grosserror,candidate,total,POSCANDIDATE,POSSOURCE,taustft,taucsp):
    P=np.asarray(P)
    POSCANDIDATE=np.asarray(POSCANDIDATE)
    taustft=np.asarray(taustft)
    pos=[]
    pot=[]
    for i in range(0,l):
        calcolaFunzioneCosto(P,POSCANDIDATE,fi,taucsp,taustft,i)
        c=np.argmin(P[i])
        pot.append(P[i][c])
        probsource.append(c)
        pos.append(POSCANDIDATE[c])
        dis=np.sqrt(np.power(POSSOURCE[i][0]-POSCANDIDATE[probsource[i]][0],2)+np.power(POSSOURCE[i][1]-POSCANDIDATE[probsource[i]][1],2))
        if(POSSOURCE[i][0]!=0):
            if int(round(dis)) <= 500:
                candidate=somma(candidate,1)
                fineerror.append(dis)
            else:
                grosserror.append(dis)
            total=somma(total,1)
    pot=np.asarray(pot)
    pos=np.asarray(pos)

    return total, candidate,pot,pos

def getCostFunctionTraining(l,fi,probsource,P,fineerror,grosserror,candidate,total,POSCANDIDATE,POSSOURCE,taustft,taucsp):
    P=np.asarray(P)
    POSCANDIDATE=np.asarray(POSCANDIDATE)
    taustft=np.asarray(taustft)
   # pos=[]
    pot=[]
    for i in range(0,l):
        calcolaFunzioneCosto(P,POSCANDIDATE,fi,taucsp,taustft,i)
        #c=np.argmin(P[i])
        #pot.append(P[i][c])
        #probsource.append(c)
        #pos.append(POSCANDIDATE[c])
        #dis=np.sqrt(np.power(POSSOURCE[i][0]-POSCANDIDATE[probsource[i]][0],2)+np.power(POSSOURCE[i][1]-POSCANDIDATE[probsource[i]][1],2))
        #if(POSSOURCE[i][0]!=0):
        #    if int(round(dis)) <= 500:
        #        candidate=somma(candidate,1)
        #        fineerror.append(dis)
        #    else:
        #        grosserror.append(dis)
        #    total=somma(total,1)
    #pot=np.asarray(pot)
    #pos=np.asarray(pos)
        if(POSSOURCE[i][0]!=0) and (POSSOURCE[i][1]!=0) and (POSSOURCE[i][2]!=0): 
            listofcost=[] 
            for n in range(0,len(POSCANDIDATE)):
                dis=np.sqrt(np.power(POSSOURCE[i][0]-POSCANDIDATE[n][0],2)+np.power(POSSOURCE[i][1]-POSCANDIDATE[n][1],2))
                if int(round(dis)) <= 500:
                    listofcost.append(P[i][n])
            pot.append(np.min(listofcost))
        else:
            pot.append(float(1))
            #pos.append([0,0,0])
    pot=np.asarray(pot)
    #pos=np.asarray(pos)
    return pot



#@jit(nopython=True)
def calcolaFunzioneCosto(P,POSCANDIDATE,fi,taucsp,taustft,i):
    for n in range(0,len(POSCANDIDATE)):
        for j in range(0,len(fi)):
            P[i][n] =(P[i][n] +np.power(somma(taustft[n][j],-taucsp[i][j]) ,2))    

#@jit(nopython=True)
def organizeGCC(GCCTEMP,fi,numcsp,l):
    GCC_feature=np.zeros(shape=(l,len(fi),numcsp))
    for i in range(len(GCCTEMP)):
        for ii in range(len(fi)):
            III=GCCTEMP[i][ii]
            IIItemp=np.concatenate((III[-int(numcsp/2):],III[0:int(numcsp/2)+1]))
            IIItemp=IIItemp/np.max(IIItemp)
            GCC_feature[i,ii]=IIItemp
            #print str(i)+' '+str(ii)
            #print GCC_feature[i,ii]
    return GCC_feature

def organizeGCCNN(GCCTEMP,fi,numcsp,l):
    GCC_feature=np.zeros(shape=(l,len(fi),numcsp))
    for i in range(len(GCCTEMP)):
        for ii in range(len(fi)):
            III=GCCTEMP[i][ii]
            IIItemp=np.concatenate((III[-int(numcsp/2):],III[0:int(numcsp/2)+1]))
            #IIItemp=IIItemp/np.max(IIItemp)
            GCC_feature[i,ii]=IIItemp
            #print str(i)+' '+str(ii)
            #print GCC_feature[i,ii]
    return GCC_feature