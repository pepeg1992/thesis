import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten,Input, Conv2D, MaxPooling2D,Conv1D,Concatenate,BatchNormalization
from keras.optimizers import Adam 
from constants import *
from keras import backend as K
import theano.tensor

def getModelSingleChannel(VAD_Kernels_N,Kernels_Size,VAD_Dense_Layers,numstrides,fi,context,VAD_tr_GCC,SLOC_tr_GCC,pathsave,nameplace):
    Weights_Init="normal"          
    SLOC_CNN_NAME="SLOC_CNN_"
    SLOC_INPUT_LAYER1=Input(shape=(len(fi),context,kmax),name='INPUT_ARRAY')
    for i in range(0, len( VAD_Kernels_N )):
        if i == 0:
            VAD_Conv_Input_GCC=SLOC_INPUT_LAYER1   
        elif i>0:
            VAD_Conv_Input_GCC = VAD_Conv1;  
        temp_name ='INPUT_ARRAY' +SLOC_CNN_NAME + str(i);
        VAD_Conv1 = Conv2D(VAD_Kernels_N[i], (Kernels_Size, Kernels_Size),strides=(numstrides,numstrides),kernel_regularizer=keras.regularizers.l1_l2(0.00001),kernel_initializer=Weights_Init,padding='same', activation='relu', name = temp_name)(VAD_Conv_Input_GCC)
    temp_name = 'INPUT_ARRAY_VAD_Flattening'
    VAD_Conv_Flattened = Flatten(name=temp_name)(VAD_Conv1)
    for i in range(0,len(VAD_Dense_Layers)):
        if i==0:
            SLOC_Dense=Dense(VAD_Dense_Layers[i],name='SLOC_DENSE'+str(int(i)),activation='relu')(VAD_Conv_Flattened)  
        else:
            SLOC_Dense=Dense(VAD_Dense_Layers[i],name='SLOC_DENSE1'+str(int(i)),activation='relu')(SLOC_Dense)
    temp_name='SLOC_OUTPUT'
    SLOC_Output_Layer_X=Dense(2,activation='relu',name=temp_name)(SLOC_Dense)              
    model = Model(inputs=[SLOC_INPUT_LAYER1], outputs =[SLOC_Output_Layer_X])
    print model.summary()
    numEpochs=500
    Batch_Size=200
    Learning_Rate=0.0001
    adam = Adam(lr= Learning_Rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam,loss='mse',metrics=['mse'])
    ES = EarlyStopping( monitor = 'loss' ,min_delta=0.0000001, patience = 5 ,verbose = 2 )
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=0.000000001,verbose=2)
    model.fit([VAD_tr_GCC],[SLOC_tr_GCC], epochs=numEpochs,batch_size=Batch_Size,verbose=2,callbacks=[ES,reduce_lr],validation_split=0.1)
    neuralnetworkpathRoom=pathsave+nameplace+'_cnntemp.h5'
    model.save(neuralnetworkpathRoom)
    return model




def getModelMultiChannel(SLOC_Kernels_N,Kernels_Size,SLOC_Dense_Layers,numstrides,fi,context,VAD_tr_GCC,SLOC_tr_GCC,pathsave,nameplace):
    Weights_Init="normal"
    SLOC_post_CNN_Dropout=[]
    SLOC_post_CNN_Dense_Layers=[]              
    SLOC_CNN_NAME="SLOC_CNN_"
    SLOC_Input = "SLOC_Input"
    SLOC_DENSE_NAME = "SLOC_DENSE_"
    SLOC_post_CNN_DENSE_NAME = "SLOC_post_CNN_DENSE_"
    SLOC_Output = "SLOC_Output"
    SLOC_post_CNN_Batch_Normalization=False
    SLOC_branch_list_input = [];
    SLOC_branch_list_flattened = [];
    SLOC_Dropout=[]
    SLOC_Activation="relu"
    N_GCC=len(fi)
    Shared_SLOC_Conv_List = [];
    Shared_SLOC_Pool_List = [];
    SLOC_Batch_Normalization=False
    for i in range(0, len( SLOC_Kernels_N )):
        Shared_SLOC_Conv = Conv1D(SLOC_Kernels_N[i], Kernels_Size,strides=numstrides, padding='same',kernel_regularizer=keras.regularizers.l1_l2(0.00001), activation=SLOC_Activation);
        Shared_SLOC_Conv_List.append(Shared_SLOC_Conv);
    for kk in range(0, N_GCC ):
        this_SLOC_Input = SLOC_Input + "_branch_" + str(kk);
        SLOC_Input_Layer = Input(shape = ( context, kmax,), name = this_SLOC_Input);
        SLOC_branch_list_input.append( SLOC_Input_Layer);
        for i in range(0, len( SLOC_Kernels_N )):
            if i == 0:
                SLOC_Conv_Input = SLOC_Input_Layer;
            elif i>0:
                SLOC_Conv_Input = SLOC_Conv;
            SLOC_Conv = Shared_SLOC_Conv_List[i](SLOC_Conv_Input)
        SLOC_Dense = None;
        temp_name = "SLOC_Flattening"
        SLOC_Conv_pre_Flattened = None;
        SLOC_Conv_pre_Flattened = Flatten()(SLOC_Conv)
        for i in range(0, len(SLOC_post_CNN_Dense_Layers)):
            if i == 0:
                SLOC_post_CNN_Dense_Input = SLOC_Conv_pre_Flattened;
            elif i>0:
                SLOC_post_CNN_Dense_Input = SLOC_post_CNN_Dense;        
            temp_name = SLOC_post_CNN_DENSE_NAME + str(i) + "_branch_" + str(kk);
            SLOC_post_CNN_Dense = Dense(SLOC_post_CNN_Dense_Layers[i], activation=SLOC_Activation, name = temp_name)(SLOC_post_CNN_Dense_Input)
            if ( len(SLOC_post_CNN_Dropout) != 0):
                temp_name = "SLOC_post_CNN_DENSE_DROPOUT" + str(i) + "_branch_" + str(kk);
                SLOC_post_CNN_Dense = Dropout(rate = SLOC_post_CNN_Dropout[i], name = temp_name)(SLOC_post_CNN_Dense)
            if (SLOC_post_CNN_Batch_Normalization == True):
                temp_name = "SLOC_post_CNN_DENSE_BN_" + str(i) + "_branch_" + str(kk);
                SLOC_post_CNN_Dense = BatchNormalization( name = temp_name)(SLOC_post_CNN_Dense)
        if SLOC_post_CNN_Dense_Layers == []:
            SLOC_branch_list_flattened.append( SLOC_Conv_pre_Flattened )
        if SLOC_post_CNN_Dense_Layers != []:            
            SLOC_branch_list_flattened.append( SLOC_post_CNN_Dense )
    for i in range(0, len(SLOC_Dense_Layers)):
        if N_GCC != 1:
            if i == 0:
                SLOC_Dense_Input = Concatenate()(SLOC_branch_list_flattened);
        if N_GCC == 1:
            if i == 0:
                SLOC_Dense_Input = SLOC_branch_list_flattened[0];
        elif i>0:
            SLOC_Dense_Input = SLOC_Dense;        
        temp_name = SLOC_DENSE_NAME + str(i);
        SLOC_Dense = Dense(SLOC_Dense_Layers[i], activation=SLOC_Activation, name = temp_name)(SLOC_Dense_Input)
        if ( len(SLOC_Dropout) != 0):
            temp_name = "SLOC_DENSE_DROPOUT" + str(i);
            SLOC_Dense = Dropout(rate = SLOC_Dropout[i], name = temp_name)(SLOC_Dense)
        if (SLOC_Batch_Normalization == True):
            temp_name = "SLOC_DENSE_BN_" + str(i);
            SLOC_Dense = BatchNormalization( name = temp_name)(SLOC_Dense)         
    temp_name = SLOC_Output;
    SLOC_Output_Layer = Dense(2, activation=SLOC_Activation, name = SLOC_Output)(SLOC_Dense)
    model = Model(inputs=SLOC_branch_list_input, outputs =SLOC_Output_Layer)
    Learning_Rate=0.0001
    adam = Adam(lr= Learning_Rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam,loss='mse',metrics=['mse'])
    print model.summary()
    ES = EarlyStopping( monitor = 'loss' ,min_delta=0.0000001, patience = 20 ,verbose = 2 )
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000000001,verbose=2)
    numEpochs=500
    batchSize=200
    model.fit(VAD_tr_GCC,[SLOC_tr_GCC], epochs=numEpochs,batch_size=batchSize,verbose=2,callbacks=[ES,reduce_lr],validation_split=0.1)
    neuralnetworkpathRoom=pathsave+nameplace+'_cnntemp.h5'
    model.save(neuralnetworkpathRoom)
    return model

def getModelJoint(VAD_Kernels_N,Kernels_Size,VAD_Dense_Layers,DropoutLayer,numstrides,fi,context,mic_log_mel,VAD_tr_LOG,VAD_tr_GCC,SLOC_tr_LOG,SLOC_tr_GCC,pathsave,nameplace):
    def hard_tanh(x, alpha=1, max_value=1.0, min_value=-1.0):
        if max_value is not None:
            x = theano.tensor.minimum(x, max_value)
        if min_value is not None:
            x = theano.tensor.maximum(x, min_value)
        return x
    SLOC_Activation_Name = "hard_tanh"                
    Joint_Activation = hard_tanh;
    keras.activations.hard_tanh = hard_tanh
    Weights_Init="normal"
    SLOC_CNN_NAME="SLOC_CNN_"
    VAD_CNN_NAME="VAD_CNN_"
    SLOC_INPUT_LAYER1=Input(shape=(len(fi),context,kmax),name='SLOC_INPUT_ARRAY')
    VAD_INPUT_LAYER1=Input(shape=(mic_log_mel,context,numfilter),name='VAD_INPUT_ARRAY')
    for i in range(0, len( VAD_Kernels_N )):
        if i == 0:
            SLOC_Conv_Input_GCC=SLOC_INPUT_LAYER1 
            VAD_Conv_Input_LOG=VAD_INPUT_LAYER1   
    
        elif i>0:
            VAD_Conv_Input_LOG = VAD_Conv1  
            SLOC_Conv_Input_GCC =SLOC_Conv1
        temp_name ='INPUT_ARRAY' +SLOC_CNN_NAME + str(i);
        VAD_Conv1 = Conv2D(VAD_Kernels_N[i], (Kernels_Size, Kernels_Size),strides=(numstrides,numstrides),kernel_regularizer=keras.regularizers.l1_l2(0.00001),kernel_initializer=Weights_Init,padding='same', activation=Joint_Activation, name = temp_name)(VAD_Conv_Input_LOG)
        temp_name='INPUT_ARRAY'+VAD_CNN_NAME+str(i)
        SLOC_Conv1 = Conv2D(VAD_Kernels_N[i], (Kernels_Size, Kernels_Size),strides=(numstrides,numstrides),kernel_regularizer=keras.regularizers.l1_l2(0.00001),kernel_initializer=Weights_Init,padding='same', activation=Joint_Activation, name = temp_name)(SLOC_Conv_Input_GCC)
    temp_name = 'INPUT_ARRAY_VAD_Flattening'
    VAD_Conv_Flattened1 = Flatten(name=temp_name)(VAD_Conv1)
    temp_name='INPUT_ARRAY_SLOC_Flattening'
    SLOC_Conv_Flattened1 = Flatten(name=temp_name)(SLOC_Conv1)
    VAD_Conv_Flattened=keras.layers.concatenate([VAD_Conv_Flattened1,SLOC_Conv_Flattened1], axis=-1)
    for i in range(0,len(VAD_Dense_Layers)):
        if i==0:
            VAD_Dense=Dense(VAD_Dense_Layers[i],name='SLOC_DENSE'+str(int(i)),activation=Joint_Activation)(VAD_Conv_Flattened)  
        else:
            VAD_Dense=Dense(VAD_Dense_Layers[i],name='SLOC_DENSE1'+str(int(i)),activation=Joint_Activation)(VAD_Dense)
        if len(DropoutLayer)!=0:
            if len(DropoutLayer)<i+1:
                VAD_Dense=Dropout(DropoutLayer[i])(VAD_Dense)
            
    temp_name='SLOC_OUTPUT'
    SLOC_Output_Layer_X=Dense(2,activation=Joint_Activation,name=temp_name+'_X')(VAD_Dense)  
    temp_name='VAD_OUTPUT'
    VAD_Output_Layer_X=Dense(2,activation='softmax',name=temp_name+'_X')(VAD_Dense)              
    model = Model(inputs=[VAD_INPUT_LAYER1,SLOC_INPUT_LAYER1], outputs =[VAD_Output_Layer_X,SLOC_Output_Layer_X])
    print model.summary()
    Learning_Rate=0.00005
    adam = Adam(lr= Learning_Rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam,loss='mse',metrics=['mse','accuracy'])
    ES = EarlyStopping( monitor = 'loss' ,min_delta=0.00001, patience = 5 ,verbose = 2 )
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=0.00000001,verbose=2)
    numEpochs=500
    model.fit([VAD_tr_LOG,VAD_tr_GCC],[SLOC_tr_LOG,SLOC_tr_GCC], epochs=numEpochs,batch_size=200,verbose=2,callbacks=[ES,reduce_lr],validation_split=0.1)
    neuralnetworkpathRoom=pathsave+nameplace+'_cnntemp.h5'
    model.save(neuralnetworkpathRoom)
    return model