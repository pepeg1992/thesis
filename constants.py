dirArray='Array'
dirWall='Wall'
formataudio='.wav'
formattext='.txt'
formatref='.ref'
formatnet='.h5'
pathdataset='/media/giovanni/TrekStor/DATASET/'
pathdir_hscma=pathdataset+'HSCMA/'
pathdir_fittizio=pathdataset+'DLS/'
pathdir_evalita=pathdataset+'DIRHA_DATASET/'
dirSource2='Additional_info'
pathdir=pathdir_evalita+'AUDIO_FILES/'
real_sim_array=['Real','Simulations']
dir_real_sim_array=['real','sim']
rate=16000
lenframe=960
hopframe=800
N=960
kmax=51
numfilter=41
lun=2*N
import numpy as np
window=np.hanning(N)
