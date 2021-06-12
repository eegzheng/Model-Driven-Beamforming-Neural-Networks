# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 17:50:09 2018
@author: wehchao xia
"""
## tensorflow version <2, install keras separately
# import keras
# from keras.models import Model
# from keras.models import load_model
# from keras.layers import Dense,Dropout,Flatten,Conv2D,Input,BatchNormalization,Activation
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.io as sio 
# from keras.callbacks import EarlyStopping, ModelCheckpoint

# tensorflow version >2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense,Flatten,Conv1D,Conv2D,Input,BatchNormalization,Activation,Dropout, Lambda,Concatenate, PReLU
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard
 
#-------------prepare training data and test data----------------------------------------
model_name='./model/large_a2u2p30dbm.h5'
test_file_name='./test_large_a2u2p30dbm.mat'
train_file_name='./train_large_a2u2p30dbm.mat'
train_data=sio.loadmat(train_file_name) 
train_sample_num=int(train_data['sample_num'])
bs_antenna_num=int(train_data['bs_antenna_num'])
user_num=int(train_data['user_num'])
train_sample_dl_power=train_data['dl_power']
train_sample_ul_power=train_data['ul_power']
train_sample_power_constr=train_data['power_constr']
train_sample_channel_complex=train_data['channel_bs_user_complex']
train_sample_channel=np.zeros((train_sample_num,1,user_num*bs_antenna_num,2))
for sample_index in range(train_sample_num):
    train_sample_channel[sample_index,0,:,0] = np.transpose(train_sample_channel_complex[:, :, sample_index]).reshape(-1,user_num*bs_antenna_num).real.astype('float32')
    train_sample_channel[sample_index,0,:,1] = np.transpose(train_sample_channel_complex[:, :, sample_index]).reshape(-1,user_num*bs_antenna_num).imag.astype('float32')

x_train= train_sample_channel    
y_train= np.transpose(np.concatenate((train_sample_ul_power,train_sample_dl_power),axis=0))/train_sample_power_constr


#----------------------construct the learning network--------------------------
inputs = Input(shape=(1,bs_antenna_num*user_num,2))
x = Conv2D(filters=8,
                name='conv1',
                kernel_size=(3,3),
                padding='same',)(inputs)
x=BatchNormalization(name='bn1')(x)
x=Activation('relu')(x)
x=Dropout(0.3)(x)
x = Conv2D(filters=8,
                name='conv2',
                kernel_size=(3,3),
                padding='same',)(x)
x=BatchNormalization(name='bn2')(x)
x=Activation('relu')(x)
x=Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(2*user_num,name='dense1')(x)
predictions=Activation('sigmoid')(x)
model=Model(inputs=inputs, outputs=predictions)
print('Model Summary',model.summary())
adamm = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
model.compile(loss='mse',optimizer=adamm,metrics=['mae'])
checkpoint = ModelCheckpoint(model_name, 
                                monitor='val_loss', 
                                verbose=1, 
                                save_best_only=True, 
                                save_weights_only=False, 
                                mode='auto', 
                                period=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)
#---------------Start training----------------------
train_history=model.fit(x=x_train,y=y_train,validation_split=0.2,epochs=10,batch_size=200,verbose=1,callbacks=[early_stopping,checkpoint])

#==============================================================================#
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
#=========================================================================#
  
show_train_history(train_history,'loss','val_loss')
# show_train_history(train_history,'mean_absolute_error','val_mean_absolute_error')
#====================================================================#testing
test_data=sio.loadmat(test_file_name) 
test_sample_num=int(test_data['sample_num'])
test_sample_channel_complex=test_data['channel_bs_user_complex']
test_sample_channel=np.zeros((test_sample_num,1,bs_antenna_num*user_num,2))
for sample_index in range(test_sample_num):
    test_sample_channel[sample_index,0,:,0] = np.transpose(test_sample_channel_complex[:,:,sample_index]).reshape(-1,user_num*bs_antenna_num).real.astype('float32')
    test_sample_channel[sample_index,0,:,1] = np.transpose(test_sample_channel_complex[:,:,sample_index]).reshape(-1,user_num*bs_antenna_num).imag.astype('float32')
test_sample_dl_beam=test_data['beam']
test_sample_ul_power=test_data['ul_power']
test_sample_dl_power=test_data['dl_power']
test_sample_sumrate=test_data['sumrate']
test_sample_power_constr=test_data['power_constr']
test_sample_noise_variance=test_data['noise_variance']

x_test=test_sample_channel
y_test=np.transpose(np.concatenate((test_sample_ul_power,test_sample_dl_power),axis=0))/test_sample_power_constr

model=load_model(model_name)  
score = model.evaluate(x_test, y_test, batch_size=200, verbose=1)
print('Test loss:', score[0])
print('Test mae:', score[1])

#==============================prediction===========================================# 
prediction=model.predict(x_test)
predict_ul_power=np.zeros((user_num,test_sample_num))
predict_dl_power=np.zeros((user_num,test_sample_num))
for sample_index in range(test_sample_num):
    factor=test_sample_power_constr/np.linalg.norm(prediction[sample_index,0:user_num],1)
    predict_ul_power[:,sample_index]=factor*np.transpose(prediction[sample_index,0:user_num])
    factor=test_sample_power_constr/np.linalg.norm(prediction[sample_index,user_num:2*user_num],1)
    predict_dl_power[:,sample_index]=factor*np.transpose(prediction[sample_index,user_num:2*user_num])
    
predict_beam_dl=np.zeros((bs_antenna_num,user_num,test_sample_num),dtype=complex)
predict_sumrate=np.zeros((1,test_sample_num))
for sample_index in  range(test_sample_num):
    TT=test_sample_noise_variance*np.eye(bs_antenna_num)
    for user_index in range(user_num):
        TT=TT+predict_ul_power[user_index,sample_index]*(np.dot(np.conj(test_sample_channel_complex[:,user_index,sample_index].reshape(-1,1)),test_sample_channel_complex[:,user_index,sample_index].reshape(1,-1)))

    for user_index in range(user_num):
        beam_up_temp=np.dot(np.linalg.inv(TT),np.conj(test_sample_channel_complex[:,user_index, sample_index]))
        predict_beam=beam_up_temp/np.linalg.norm(beam_up_temp)
        predict_beam_dl[:,user_index,sample_index]=np.sqrt(predict_dl_power[user_index,sample_index])*predict_beam
       
    for user_index in range(user_num):
        sig=np.square(np.linalg.norm(predict_beam_dl[:,user_index,sample_index].dot(test_sample_channel_complex[:,user_index,sample_index])))
        interf=test_sample_noise_variance-sig+np.square(np.linalg.norm(test_sample_channel_complex[:,user_index,sample_index].dot(predict_beam_dl[:,:,sample_index])))
        predict_sumrate[0,sample_index]=predict_sumrate[0,sample_index]+np.log2(1+sig/interf)

diff=0
for sample_index in range(test_sample_num):
    diff=diff+np.abs(test_sample_sumrate[0,sample_index]-predict_sumrate[0,sample_index])/test_sample_sumrate[0,sample_index]
ave_diff=diff/test_sample_num
print('ave_diff:',ave_diff)
    
ave_predict_sumrate=np.sum(predict_sumrate)/test_sample_num
ave_real_sumrate=np.sum(test_sample_sumrate)/test_sample_num
print('ave_predict_sumrate:',ave_predict_sumrate)
print('ave_real_sumrate:',ave_real_sumrate)

diff2=(np.sum(predict_sumrate)-np.sum(test_sample_sumrate))/test_sample_num
print('diff2:',diff2)




