# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 17:50:09 2018
@author: wehchao xia
"""
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
train_data=sio.loadmat('./train_large_complex_a4u4p20dbm.mat') 
train_sample_num=int(train_data['sample_num'])
bs_antenna_num=int(train_data['bs_antenna_num'])
user_num=int(train_data['user_num'])
train_sample_ul_power=train_data['power_uplink']
train_sample_power_constr=train_data['power_constr']
train_sample_channel_complex=train_data['channel_bs_user_complex']
#train_sample_channel=np.concatenate((train_sample_channel_complex.real,train_sample_channel_complex.imag),axis=1)
train_sample_channel=np.zeros((train_sample_num,2,user_num*bs_antenna_num,1))
for sample_index in range(train_sample_num):
    train_sample_channel[sample_index,0,:,0] = np.transpose(train_sample_channel_complex[:, :, sample_index]).reshape(-1,user_num*bs_antenna_num).real.astype('float32')
    train_sample_channel[sample_index,1,:,0] = np.transpose(train_sample_channel_complex[:, :, sample_index]).reshape(-1,user_num*bs_antenna_num).imag.astype('float32')

x_train= train_sample_channel    
y_train= np.transpose(train_sample_ul_power)/train_sample_power_constr


#----------------------construct the learning network--------------------------
    
inputs = Input(shape=(2,bs_antenna_num*user_num,1))
x = Conv2D(filters=8,
                kernel_size=(3,3),
                padding='same',
                activation='relu')(inputs)
x = Conv2D(filters=8,
                kernel_size=(3,3),
                padding='same',#补零
                activation='relu')(x) 
x = Flatten()(x)
predictions = Dense(user_num,activation='sigmoid')(x)
model=Model(inputs=inputs, outputs=predictions)

print('Model Summary',model.summary())
adamm = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)
model.compile(loss='mse',optimizer=adamm,metrics=['mae'])
checkpoint = ModelCheckpoint('./model/large_a4u4p20dbm.h5', 
                                monitor='val_loss', 
                                verbose=1, 
                                save_best_only=True, 
                                save_weights_only=False, 
                                mode='auto', 
                                period=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)
#---------------Start training----------------------
train_history=model.fit(x=x_train,y=y_train,validation_split=0.2,epochs=100,batch_size=200,verbose=1,callbacks=[early_stopping,checkpoint])

#==============================================================================#
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
   
show_train_history(train_history,'loss','val_loss') 

 

test_data=sio.loadmat('./test_large_complex_a4u4p20dbm.mat') 
test_sample_num=int(test_data['sample_num'])
test_sample_channel_complex=test_data['channel_bs_user_complex']
test_sample_channel=np.zeros((test_sample_num,2,bs_antenna_num*user_num,1))
for sample_index in range(test_sample_num):
    test_sample_channel[sample_index,0,:,0] = np.transpose(test_sample_channel_complex[:,:,sample_index]).reshape(-1,user_num*bs_antenna_num).real.astype('float32')
    test_sample_channel[sample_index,1,:,0] = np.transpose(test_sample_channel_complex[:,:,sample_index]).reshape(-1,user_num*bs_antenna_num).imag.astype('float32')
test_sample_dl_beam=test_data['dl_beam']
test_sample_ul_beam=test_data['ul_beam']
test_sample_ul_power=test_data['power_uplink']
test_sample_sinr=test_data['sinr']
test_sample_power_constr=test_data['power_constr']
test_sample_noise_variance=test_data['noise_variance']
test_sample_SINR_target=test_data['SINR_target']

x_test=test_sample_channel
y_test=np.transpose(test_sample_ul_power)/test_sample_power_constr
score = model.evaluate(x_test, y_test, batch_size=200, verbose=1)

print('Test loss:', score[0])
print('Test mae:', score[1])

#================================================================================= predict
prediction=model.predict(x_test)
predict_ul_power=np.zeros((user_num,test_sample_num))
for sample_index in range(test_sample_num):
    factor=test_sample_power_constr/np.linalg.norm(prediction[sample_index,:],1)
    predict_ul_power[:,sample_index]=factor*np.transpose(prediction[sample_index,:])
    
predict_ul_beam=np.zeros((bs_antenna_num,user_num,test_sample_num),dtype=complex)
beam_dl=np.zeros((bs_antenna_num,user_num,test_sample_num),dtype=complex)

sinr_dl=np.zeros((user_num,test_sample_num))
power_dl=np.zeros((user_num,test_sample_num))

for sample_index in  range(test_sample_num):
    TT=test_sample_noise_variance*np.eye(bs_antenna_num)
    for user_index in range(user_num):
        TT=TT+predict_ul_power[user_index,sample_index]*(np.dot(np.conj(test_sample_channel_complex[:,user_index,sample_index].reshape(-1,1)),test_sample_channel_complex[:,user_index,sample_index].reshape(1,-1)))

    for user_index in range(user_num):
        beam_up_temp=np.dot(np.linalg.inv(TT),np.conj(test_sample_channel_complex[:,user_index, sample_index]))
        predict_ul_beam[:,user_index,sample_index]=beam_up_temp/np.linalg.norm(beam_up_temp)
       
 
    D=np.zeros((user_num,user_num))
    for user_index in range(user_num):
        D[user_index,user_index]=test_sample_SINR_target[user_index]/(np.square(np.linalg.norm(np.conj(predict_ul_beam[:,user_index,sample_index]).dot(np.conj(test_sample_channel_complex[:,user_index,sample_index]))))/test_sample_noise_variance)
        
    F=np.zeros((user_num,user_num))
    for user_index in range(user_num):
        for  kk in range(user_num):
            if kk != user_index:
                F[user_index,kk]=np.square(np.linalg.norm(np.conj(predict_ul_beam[:,kk,sample_index]).dot(np.conj(test_sample_channel_complex[:,user_index,sample_index]))))/test_sample_noise_variance
            else:
                F[user_index,kk]=0    
        
    #downlink
    u=np.ones((1,user_num))
    de=np.ones((user_num,1))
    X=np.vstack((np.hstack((D.dot(F), D.dot(de))),np.hstack((u.dot(D).dot(F)/test_sample_power_constr, u.dot(D).dot(de)/test_sample_power_constr))))
    a,b=np.linalg.eig(X)#a eigenvalue ,b eignvector
    la2=np.amax(np.real(a))
    index=np.argmax(np.real(a))
    ptemp=b[:,index]
    pptemp=ptemp/ptemp[user_num]
    power_dl[:,sample_index]=np.real(pptemp[:user_num])
    
    for user_index in range(user_num):
        beam_dl[:,user_index,sample_index]=np.sqrt(power_dl[user_index,sample_index])*predict_ul_beam[:,user_index,sample_index]
    
    for user_index in range(user_num):
        sig=np.square(np.linalg.norm(beam_dl[:,user_index,sample_index].dot(test_sample_channel_complex[:,user_index,sample_index])))
        interf=test_sample_noise_variance-sig+np.square(np.linalg.norm(test_sample_channel_complex[:,user_index,sample_index].dot(beam_dl[:,:,sample_index])))
        sinr_dl[user_index,sample_index]=sig/interf

diff=0
for sample_index in range(test_sample_num):
    diff=diff+np.abs(np.amin(test_sample_sinr[:,sample_index])-np.amin(sinr_dl[:,sample_index]))/np.amin(test_sample_sinr[:,sample_index])
ave_diff=diff/test_sample_num
print('Averge difference:',ave_diff)
    






