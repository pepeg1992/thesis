import numpy as np
import scipy
from scipy.io import wavfile

pathdir='/home/giovanni/DIRHA_DATASET/'
pathsave=pathdir+'VADSLOC50/'
dirSource2='Additional_info'
dirArray='Array'
dirWall='Wall'
formataudio='.wav'
formattext='.txt'
formatref='.ref'

def positionMic(filemic):

    x=[]
    y=[]
    z=[]
    for i in range(0,len(filemic)):
        f = open(filemic[i],'r')
        foo=f.read()
        xstring=foo.find('x=')
        pointxstring=foo.find(';',xstring)
        ystring=foo.find('y=')
        pointystring=foo.find(';',ystring)
        zstring=foo.find('z=')
        pointzstring=foo.find(';',zstring)
        xst=foo[xstring+2:pointxstring]
        yst=foo[ystring+2:pointystring]
        zst=foo[zstring+2:pointzstring]
        x.append(float(xst)*10.0)
        y.append(float(yst)*10.0)
        z.append(float(zst)*10.0)
    POSMIC = np.c_[x, y, z]
    return POSMIC

def positionSource(filesource):
    f = open(filesource,'r')
    foo=f.readline()
    x=[]
    y=[]
    z=[]
    for i,line in enumerate(f):
        if 'sp' in line and not 'spoon' in line:
            l=line.find('#')
            m=line.find('sp')
            if m<l:
                m=line.find(' ', m)
                n=line.find(' ', m+1)
                xst=line[m+1:n]
                m=line.find(' ',n+1)
                yst=line[n+1:m]
                n=line.find(' ',m+1)
                zst=line[m+1:n]
            else:
                xst='0'
                yst='0'
                zst='0'

        else:
            xst='0'
            yst='0'
            zst='0'
        x.append(float(xst))
        y.append(float(yst))
        z.append(float(zst))

    POSSOURCE = np.c_[x, y, z]
    return POSSOURCE


def positionCansim(filesource):
    f = open(filesource,'r')
    foo=f.readline()
    x=[]
    y=[]
    z=[]
    for i,line in enumerate(f):
        l=line.find('#')
           
        if l>22:
            m=line.find(' ', 14)
            n=line.find(' ', m+1)
            xst=line[m+1:n]
            m=line.find(' ',n+1)
            yst=line[n+1:m]
            n=line.find(' ',m+1)
            zst=line[m+1:n]
        else:
            xst='0'
            yst='0'
            zst='0'

        x.append(float(xst))
        y.append(float(yst))
        z.append(float(zst))
    eliminate=[]
    for i in range(0,len(x)-1):
    
        if (x[i]!=0.0):
            for k in range(i+1,len(x)):
                if(x[i] == x[k] and y[i] == y[k]) and x[k]!=0 :
                    eliminate.append(k)
        else:
            eliminate.append(i)
    
    xs=np.delete(x,eliminate)
    ys=np.delete(y,eliminate)
    zs=np.delete(z,eliminate)
    POSSOURCE = np.c_[xs, ys, zs]
    return POSSOURCE


def audioSource(fileaudio):
    audio=[]
    for i in range(0,len(fileaudio)):
        #print fileaudio[i]
        rate, data=scipy.io.wavfile.read(fileaudio[i])
        #print data
        #print len(data)
        audio.append(np.divide(data.astype(np.float64),2**15))
    return audio


def getCandidate(files):
    f = open(files)
    foo=f.readlines()
    x=[]
    y=[]
    z=[]
    for i in range(0,len(foo)):
        st=foo[i]
        m=st.find(' ')
        xst=st[0:m-1]
        n=st.find(' ',m+1)
        yst=st[m+1:n-1]
        m=st.find('\n',n+1)
        zst=st[n+1:m-1]
        x.append(float(xst))
        y.append(float(yst))
        z.append(float(zst))
    POSCANDIDATE = np.c_[x, y, z]
    return POSCANDIDATE


def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i

def positionSys(filesource):
    f = open(filesource,'r')
   # print filesource
    foo=f.readline()
    x=[]
    y=[]
    z=[]
    for i,line in enumerate(f):
        if ('sys' in line) or ('pro' in line):
            syscheck='1'
           # print syscheck
        else:
            syscheck='0'
        x.append(float(syscheck))

    POSSOURCE = np.c_[x]
    return x

def getMic(nameplace, dira):
    fileaudio=[]
    if nameplace=='Livingroom':
        for i in range(1, 7):
            text='LA'+str(i)
            fileaudio.append(dira+'/'+dirArray+'/'+text+formataudio)
        text='L1C'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        for i in range(1, 5):
            text='L'+str(i)+'L'
            fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
            text='L'+str(i)+'R'
            fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
    elif nameplace=='Kitchen':
        for i in range(1, 7):
            text='KA'+str(i)
            fileaudio.append(dira+'/'+dirArray+'/'+text+formataudio)
        for i in range(1, 4):
            text='K'+str(i)+'L'
            fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
            text='K'+str(i)+'R'
            fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        text='K3C'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
    elif nameplace=='Corridor':
        text='C1L'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        text='C1R'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
    elif nameplace=='Bedroom':
        text='B1L'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        text='B1R'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        text='B2C'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        text='B2L'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        text='B2R'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        text='B3L'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        text='B3R'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
    elif nameplace=='Bathroom':
        text='R1C'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        text='R1L'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
        text='R1R'
        fileaudio.append(dira+'/'+dirWall+'/'+text+formataudio)
    AUDIO = audioSource(fileaudio)
    return AUDIO

def positionSource16(filesource):
    f = open(filesource,'r')
    foo=f.readline()
    x=[]
    y=[]
    z=[]
    for i,line in enumerate(f):
        l=line.find('\n')
        m=0
        if m<l:
            #print m
            #m=line.find(' ', m)
            #print m
            n=line.find(' ', m+1)
            #print 'm: '+str(m)+'\tn: '+str(n)
            xst=line[0:4]
            m=line.find(' ',n+1)
            #print 'm: '+str(m)+'\tn: '+str(n)
            yst=line[5:9]
            n=line.find(' ',m+1)
            #print 'm: '+str(m)+'\tn: '+str(n)
            zst=line[10:14]
        x.append(float(xst))
        y.append(float(yst))
        z.append(float(zst))
    
    POSSOURCE = np.c_[x, y, z]
    return POSSOURCE

def getMic16(nameplace, dira):
    print dira
    fileaudio=[]
    if nameplace=='Livingroom':
        for i in range(1, 7):
            text='LA'+str(i)
            fileaudio.append(dira+'_'+text+formataudio)
        text='L1C'
        fileaudio.append(dira+'_'+text+formataudio)
        for i in range(1, 5):
            text='L'+str(i)+'L'
            fileaudio.append(dira+'_'+text+formataudio)
            text='L'+str(i)+'R'
            fileaudio.append(dira+'_'+text+formataudio)
    elif nameplace=='Kitchen':
        for i in range(1, 7):
            text='KA'+str(i)
            fileaudio.append(dira+'_'+text+formataudio)
        for i in range(1, 4):
            text='K'+str(i)+'L'
            fileaudio.append(dira+'_'+text+formataudio)
            text='K'+str(i)+'R'
            fileaudio.append(dira+'_'+text+formataudio)
        text='K3C'
        fileaudio.append(dira+'_'+text+formataudio)
    elif nameplace=='Corridor':
        text='C1L'
        fileaudio.append(dira+'_'+text+formataudio)
        text='C1R'
        fileaudio.append(dira+'_'+text+formataudio)
    elif nameplace=='Bedroom':
        text='B1L'
        fileaudio.append(dira+'_'+text+formataudio)
        text='B1R'
        fileaudio.append(dira+'_'+text+formataudio)
        text='B2C'
        fileaudio.append(dira+'_'+text+formataudio)
        text='B2L'
        fileaudio.append(dira+'_'+text+formataudio)
        text='B2R'
        fileaudio.append(dira+'_'+text+formataudio)
        text='B3L'
        fileaudio.append(dira+'_'+text+formataudio)
        text='B3R'
        fileaudio.append(dira+'_'+text+formataudio)
    elif nameplace=='Bathroom':
        text='R1C'
        fileaudio.append(dira+'_'+text+formataudio)
        text='R1L'
        fileaudio.append(dira+'_'+text+formataudio)
        text='R1R'
        fileaudio.append(dira+'_'+text+formataudio)
    AUDIO = audioSource(fileaudio)
    #print AUDIO
    return AUDIO


def getGrid(nameplace,pathdir,dirArray,dirWall,formattext,grid):
    
    dirMic=pathdir+'/Simulations/sim1/Signals/Mixed_Sources/'+nameplace
    dirMicArray=dirMic+'/'+dirArray
    dirMicWall=dirMic+'/'+dirWall
    fi=[]
    filemic=[]
    POSCANDIDATE=[]
    if nameplace=='Livingroom':
        print 'get position '+nameplace
        usb=[4.790,4.850,2.743]
        lsb=[0.0,0.0,0.0]
        for i in range(1, 7):
            text='LA'+str(i)
            filemic.append(dirMicArray+'/'+text+formattext)
        text='L1C'
        filemic.append(dirMicWall+'/'+text+formattext)
        for i in range(1, 5):
            text='L'+str(i)+'L'
            filemic.append(dirMicWall+'/'+text+formattext)
            text='L'+str(i)+'R'
            filemic.append(dirMicWall+'/'+text+formattext)
        POSMIC=positionMic(filemic)
        for i in range(0,6):
            for k in range(0,6):
                if i>k:
                    fi.append([i,k])
        for i in range(6,9):
            for k in range(6,9):
                if i>k:
                    fi.append([i,k])
        for i in range(9,11):
            for k in range(9,11):
                if i>k:
                    fi.append([i,k])
        for i in range(11,13):
            for k in range(11,13):
                if i>k:
                    fi.append([i,k])
        for i in range(13,15):
            for k in range(13,15):
                if i>k:
                    fi.append([i,k])
        for i in range (int(lsb[0]*1000),int(usb[0]*1000),grid):
            for j in range(int(lsb[1]*1000),int(usb[1]*1000),grid):
                for k in range(int(lsb[2]*1000),int(usb[2]*1000),grid):
                    POSCANDIDATE.append([i,j,k])
    elif nameplace=='Kitchen':
        print 'get position '+nameplace
        lsb=[0.0,0.0,0.0]
        usb=[4.790,3.800,2.743]        
        for i in range(1, 7):
            text='KA'+str(i)
            filemic.append(dirMicArray+'/'+text+formattext)
        for i in range(1, 4):
            text='K'+str(i)+'L'
            filemic.append(dirMicWall+'/'+text+formattext)
            text='K'+str(i)+'R'
            filemic.append(dirMicWall+'/'+text+formattext)
        text='K3C'
        filemic.append(dirMicWall+'/'+text+formattext)
        POSMIC=positionMic(filemic)
        for i in range(0,6):
            for k in range(0,6):
                if i>k:
                    fi.append([i,k])
        for i in range(6,8):
            for k in range(6,8):
                if i>k:
                    fi.append([i,k])
        for i in range(8,10):
            for k in range(8,10):
                if i>k:
                    fi.append([i,k])
        for i in range(10,13):
            for k in range(10,13):
                if i>k:
                    fi.append([i,k])
        for i in range (int(lsb[0]*1000),int(usb[0]*1000),grid):
            for j in range(int(lsb[1]*1000),int(usb[1]*1000),grid):
                for k in range(int(lsb[2]*1000),int(usb[2]*1000),grid):
                    POSCANDIDATE.append([i,j,k])    
    elif nameplace=='Bathroom':
        print 'get position '+nameplace
        lsb=[0.0,0.0,0.0]
        usb=[2.220,3.160,2.743]
        text='R1C'
        filemic.append(dirMicWall+'/'+text+formattext)
        text='R1L'
        filemic.append(dirMicWall+'/'+text+formattext)
        text='R1R'
        filemic.append(dirMicWall+'/'+text+formattext)
        POSMIC=positionMic(filemic)
        for i in range(0,3):
            for k in range(0,3):
                if i>k:
                    fi.append([i,k])
        for i in range (0,int(usb[0]*1000),grid):
            for j in range(0,int(usb[1]*1000),grid):
                for k in range(0,int(usb[2]*1000),grid):
                    POSCANDIDATE.append([i,j,k])    
    elif nameplace=='Corridor':
        print 'get position '+nameplace
        lsb=[0.0,0.0,0.0]
        usb=[2.550,1.770,2.743]
        text='C1L'
        filemic.append(dirMicWall+'/'+text+formattext)
        text='C1R'
        filemic.append(dirMicWall+'/'+text+formattext)
        POSMIC=positionMic(filemic)
        for i in range(0,2):
            for k in range(0,2):
                if i>k:
                    fi.append([i,k])
        for i in range (0,int(usb[0]*1000),grid):
            for j in range(0,int(usb[1]*1000),grid):
                for k in range(0,int(usb[2]*1000),grid):
                    POSCANDIDATE.append([i,j,k])
    elif nameplace=='Bedroom':
        print 'get position '+nameplace
        lsb=[0,0,0]
        usb=[2.700,4.840,2.743]
        text='B1L'
        filemic.append(dirMicWall+'/'+text+formattext)
        text='B1R'
        filemic.append(dirMicWall+'/'+text+formattext)
        text='B2C'
        filemic.append(dirMicWall+'/'+text+formattext)
        text='B2L'
        filemic.append(dirMicWall+'/'+text+formattext)
        text='B2R'
        filemic.append(dirMicWall+'/'+text+formattext) 
        text='B3L'
        filemic.append(dirMicWall+'/'+text+formattext)
        text='B3R'
        filemic.append(dirMicWall+'/'+text+formattext)       
        POSMIC=positionMic(filemic)
        for i in range(0,2):
            for k in range(0,2):
                if i>k:
                    fi.append([i,k])
        for i in range(2,5):
            for k in range(2,5):
                if i>k:
                    fi.append([i,k])
        for i in range(5,7):
            for k in range(5,7):
                if i>k:
                    fi.append([i,k])                        
        for i in range (0,int(usb[0]*1000),grid):
            for j in range(0,int(usb[1]*1000),grid):
                for k in range(0,int(usb[2]*1000),grid):
                    POSCANDIDATE.append([i,j,k])       
    POSMIC=np.asarray(POSMIC)
    POSCANDIDATE=np.asarray(POSCANDIDATE)
    fi=np.asarray(fi)
    return POSMIC,POSCANDIDATE,fi,lsb,usb

def getPhi(nameplace,pathdir,dirArray,dirWall,formattext):
        
    dirMic=pathdir+'/Simulations/sim1/Signals/Mixed_Sources/'+nameplace
    dirMicArray=dirMic+'/'+dirArray
    dirMicWall=dirMic+'/'+dirWall
    fi=[]
    filemic=[]
    POSCANDIDATE=[]
    num=[]
      
    micArray=[]
    if nameplace=='Livingroom':
        print 'get position '+nameplace
        usb=[4.790,4.850,2.743]
        lsb=[0.0,0.0,0.0]
        n=0 
        m=[]
        for i in range(0,6):
            for k in range(0,6):
                if i>k:
                    n+=1
                    m.append([i,k])

                    fi.append([i,k])
        num.append(n)

        micArray.append(m)
        n=0 
        m=[]
        for i in range(6,9):
            for k in range(6,9):
                if i>k:
                    n+=1
                    m.append([i,k])

                    fi.append([i,k])
        num.append(n)

        micArray.append(m)
        n=0 
        m=[]
        for i in range(9,11):
            for k in range(9,11):
                if i>k:
                    n+=1
                    m.append([i,k])

                    fi.append([i,k])
        num.append(n)

        micArray.append(m)        
        n=0 
        m=[]
        for i in range(11,13):
            for k in range(11,13):
                if i>k:
                    n+=1
                    m.append([i,k])

                    fi.append([i,k])
        num.append(n)

        micArray.append(m)
        n=0 
        m=[]
        for i in range(13,15):
            for k in range(13,15):
                if i>k:
                    n+=1
                    m.append([i,k])

                    fi.append([i,k])
        num.append(n)

        micArray.append(m)
    elif nameplace=='Kitchen':
        print 'get position '+nameplace
        lsb=[0.0,0.0,0.0]
        usb=[4.790,3.800,2.743]    
 

        n=0 
        m=[]
        for i in range(0,6):
            for k in range(0,6):
                if i>k:
                    n+=1
                    fi.append([i,k])
                    m.append([i,k])
        num.append(n)

        micArray.append(m)
        n=0
        m=[]
        for i in range(6,8):
            for k in range(6,8):
                if i>k:
                    fi.append([i,k])
                    n+=1
                    m.append([i,k])
        num.append(n)
        micArray.append(m)
        m=[]
        n=0
        for i in range(8,10):
            for k in range(8,10):
                if i>k:
                    fi.append([i,k])
                    m.append([i,k])
                    n+=1
        num.append(n)
        micArray.append(m)
        m=[]
        n=0
        for i in range(10,13):
            for k in range(10,13):
                if i>k:
                    fi.append([i,k])
                    m.append([i,k])
                    n+=1
        num.append(n)
        micArray.append(m)
        n=0
  
    elif nameplace=='Bathroom':
        print 'get position '+nameplace
        lsb=[0.0,0.0,0.0]
        usb=[2.220,3.160,2.743]
        for i in range(0,3):
            for k in range(0,3):
                if i>k:
                    fi.append([i,k])
        micArray=[[0,1,2]]   
    elif nameplace=='Corridor':
        print 'get position '+nameplace
        lsb=[0.0,0.0,0.0]
        usb=[2.550,1.770,2.743]
        for i in range(0,2):
            for k in range(0,2):
                if i>k:
                    fi.append([i,k])
        micArray=[[0,1]]
    elif nameplace=='Bedroom':
        print 'get position '+nameplace
        lsb=[0,0,0]
        usb=[2.700,4.840,2.743]
        for i in range(0,2):
            for k in range(0,2):
                if i>k:
                    fi.append([i,k])
        for i in range(2,5):
            for k in range(2,5):
                if i>k:
                    fi.append([i,k])
        for i in range(5,7):
            for k in range(5,7):
                if i>k:
                    fi.append([i,k])                        
    
    fi=np.asarray(fi)
    micArray=np.asarray(micArray)
    print fi
    return micArray,num,fi,lsb,usb