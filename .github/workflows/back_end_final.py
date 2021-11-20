
# coding: utf-8

# In[1]:


#Import

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import numpy as np
#import keras
import scipy as sc
from tensorflow.keras.layers import Dense,Conv2D,Conv2DTranspose,Flatten,Input,concatenate,Reshape,Dropout,Activation,BatchNormalization,MaxPooling2D,UpSampling2D
from tensorflow.keras.callbacks import CSVLogger,TerminateOnNaN,EarlyStopping,ModelCheckpoint
import scipy.io
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import csv
from tensorflow.keras.models import load_model
from pandas import *
import skfmm
from scipy.io import loadmat

import matplotlib.pyplot as plt
import matplotlib
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import glob
from PIL import Image

class SWISH(Activation):

        def __init__(self,activation,**kwargs):
                super(SWISH,self).__init__(activation,**kwargs)
                self.__name__='swish_fn'

def swish(x):
        return (x*K.sigmoid(x))

get_custom_objects().update({'swish_fn': SWISH(swish)})

tf.keras.optimizers.Adam(learning_rate=0.001,decay=0.5)

def load_data(data_file):
        
    x_list, y_list, pres_abs, x_velocity, y_velocity = np.loadtxt(data_file,
        skiprows=1,
        unpack=True)
    
    nx = 150
    ny = 150

    X_array  = np.ones((nx,ny))
    Y_array  = np.ones((nx,ny))
    P_array  = np.ones((nx,ny))
    U_array  = np.ones((nx,ny))
    V_array  = np.ones((nx,ny))

    for i in range(nx):
        for j in range(ny):         # trục y cắm xuống, trục x giữ nguyên
            if (i==0):
                X_array[i,j]     = x_list[j]
                Y_array[i,j]     = y_list[j] 
                P_array[i,j]  = pres_abs [j]/100000.      #normalize          # mấy cái này là gì
                U_array[i,j]  = x_velocity   [j]/100 
                V_array[i,j]  = y_velocity   [j]/100
            else:
                X_array[i,j]     = x_list[(j + nx*i)]
                Y_array[i,j]     = y_list[(j + nx*i)] 
                P_array[i,j]  = pres_abs [(j + nx*i)]/100000.      #normalize 
                U_array[i,j]  = x_velocity   [(j + nx*i)]/100 
                V_array[i,j]  = y_velocity   [(j + nx*i)]/100           
        
    return X_array,Y_array,P_array,U_array,V_array
#==========================================================================
#
#       Build model
#
#==========================================================================

def build_model(num_conv,num_dense,inp_optim,inp_loss):

        input1=Input(shape=(150,150,1),name='input1')
        input2=Input(shape=(2,1),name='input2')

        num_units=np.zeros(num_conv,dtype=int)
        num_kernel=np.zeros(num_conv,dtype=int)
        num_strides=np.zeros(num_conv,dtype=int)


        #------------------------------------------------------------------------------------------------------

        with open('config_file.csv','r') as csvfile:
            info = csv.reader(csvfile, delimiter=',')
            info=list(info)


        user_input_channels=int(info[0][1])

        #--------------------------------------------------------------------------------------------------------

        print("Constructing the convolutional layers:")    
 #----------------------------------------------------------------------------------Convolutional Layers Construction  
        maxpling=[]
        maxpling_size=[]
        for i in range(1,num_conv+1):
            print("Convolutional Layer %d"%(i))
            inp_numfilters   = int(info[i][1])
            num_units[i-1]   = int(info[i][1])
            inp_shape        = int(info[i][2])
            num_kernel[i-1]  = int(info[i][2])
            inp_stride       = int(info[i][3])
            num_strides[i-1] = int(info[i][3])
            inp_activation   = info[i][4]
            
            if i==1:
		
                output=Conv2D(inp_numfilters,(inp_shape,inp_shape),strides=(inp_stride,inp_stride),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(input1)
                output=BatchNormalization()(output)
            else:
                output=Conv2D(inp_numfilters,(inp_shape,inp_shape),strides=(inp_stride,inp_stride),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                output=BatchNormalization()(output)
            inp_maxpling=info[i][5]
            inp_maxpling_size=int(info[i][6])
            maxpling.append(inp_maxpling)
            maxpling_size.append(inp_maxpling_size)
            if inp_maxpling=='Y':
                output=MaxPooling2D(pool_size=(inp_maxpling_size,inp_maxpling_size),strides=(inp_maxpling_size,inp_maxpling_size))(output)
                
            else:
                continue
                
 #------------------------------------------------------------------------------------       
        shape_2=output.shape
        print(shape_2)
        output=Reshape((-1,1))(output)
        output=concatenate([output,input2],axis=-2)
        output=Flatten()(output)
        shape_1=output.shape
        print(shape_1)
        
        print("Constructing the Dense Layers")
 #------------------------------------------------------------------------------------Dense Layers Construction       
        for i in range(1,num_dense+1): 
            print("Dense Layer %d"%(i))
            inp_numunits=int(info[num_conv+i][1])
            inp_activation=info[num_conv+i][2]
            
            if i==num_dense:
                output=Dense(shape_1[1]-2,activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                output=BatchNormalization()(output)
            else:
                output=Dense(inp_numunits,activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                output=BatchNormalization()(output)
        
            
 #------------------------------------------------------------------------------------
        print(output.shape)
        output=Reshape((shape_2[1],shape_2[2],shape_2[3]))(output)
        print("Creating correspondingly symmetrical Deconvolutional layers")
        
        if user_input_channels==3:
 #------------------------------------------------------------------------------------ DeConvolutional Layers Construction 
    
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output)
                if i==num_conv:
                    output=Conv2DTranspose(3,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output)
                else:
                    output=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output)
                    output=BatchNormalization()(output)
                    
#--------------------------------------------------------------------------------------------------------------------------------  
            model=Model(inputs=[input1,input2],outputs=[output])


 #--------------------------------------------------------------------------------------------------------------------------------  
 #---------------------------------------------------------------------------------------------------------------------   
 #-----------------------------------------------------------------------------------------------------------------------------
        else:
        
            output1=output
            output2=output
            output3=output
     #-----------------------------------------------------------------------------------3-path Deconvolution output       
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output1=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output1)
                if i==num_conv:
                    output1=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output1)

                elif i==1:
                    output1=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output1)
                    output1=BatchNormalization()(output1)
                else:
                    output1=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output1)
                    output1=BatchNormalization()(output1)
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output2=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output2)
                if i==num_conv:
                    output2=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output2)
                elif i==1:
                    output2=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output2)
                    output2=BatchNormalization()(output2)
                else:
                    output2=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output2)
                    output2=BatchNormalization()(output2)
            for i in range(1,num_conv+1):
                if maxpling[num_conv-i]=='Y':
                    output3=UpSampling2D(size=(maxpling_size[num_conv-i],maxpling_size[num_conv-i]))(output3)
                if i==num_conv:
                    output3=Conv2DTranspose(1,(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation='linear',kernel_regularizer=regularizers.l2(0.00001))(output3)
                elif i==1:
                    output3=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output3)
                    output3=BatchNormalization()(output3)
                else:
                    output3=Conv2DTranspose(num_units[num_conv-i],(num_kernel[num_conv-i],num_kernel[num_conv-i]),strides=(num_strides[num_conv-i],num_strides[num_conv-i]),activation=inp_activation,kernel_regularizer=regularizers.l2(0.00001))(output3)
                    output3=BatchNormalization()(output3)
            model=Model(inputs=[input1,input2],outputs=[output1,output2,output3])  
     #-----------------------------------------------------------------------------------------------------------------------------
     #-----------------------------------------------------------------------------------------------------------------------------

        model.compile(optimizer=inp_optim,loss='mse')

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
             json_file.write(model_json)

        model.summary()
        return model
    
#=================================================================
#
#       Train model
#
#=================================================================

def train_model(batch_sz,eps,val_splt,model):
    with open('config_file_2.csv','r') as csvfile:     
      
      # lấy giá trị file config  
        info=csv.reader(csvfile,delimiter=',')         
        info=list(info)
        print('info',info)                           
    exp_no=int(info[0][0])                             
    user_input_channels=int(info[0][1])                
    user_inp=info[0][2]                                
    if user_inp=='Y':
        user_moni=info[0][3]                           
        inp_delta=float(info[0][4])                     
        inp_mineps=float(info[0][5])                    
    
 #===============================================================


        # x_train_2
        data = read_csv("./DATA_TRAINING/label_8000.csv")       
        data_size = int(8000)

        Re = data['Reynolds'].tolist()                   
        Time = data['Time'].tolist()  

        x_train_2 = np.ones((data_size,2))
        x_train_2[:,0] = Re                           
        x_train_2[:,1] = Time
        x_train_2[:,0] = x_train_2[:,0] #/100       
        x_train_2[:,1] = x_train_2[:,1] #/273              

        
        # y_train
        P_array  = []
        U_array  = []
        V_array  = []
        sdf_array= []

        for i in range(data_size):  
          data_file = './DATA_TRAINING/'+str(Re[i])+'/'+str(Re[i])+'_'+str(f"{Time[i]:04n}")+'.txt'   
          print('reading data file ...',data_file)
          x_list,y_list,pres_abs,x_velocity,y_velocity=load_data(data_file)                             
          P_array.append(pres_abs)                                    
          U_array.append(x_velocity)
          V_array.append(y_velocity)

               
        y_train_2 = np.asarray(P_array).astype('float32')                                                            
        y_train_3 = np.asarray(U_array).astype('float32')
        y_train_4 = np.asarray(V_array).astype('float32')

        
        y_train_2=np.expand_dims(y_train_2,axis=3)           
        y_train_3=np.expand_dims(y_train_3,axis=3)
        y_train_4=np.expand_dims(y_train_4,axis=3)     

        #geometry construct - SDF function
        dx = 0.49/300                                                                    
        X, Y = np.meshgrid(np.linspace(-0.15,0.34,150), np.linspace(-0.23,0.26,150))
        phi = 1 * np.ones_like(X)
        phi[np.logical_and(X**2 + Y**2 < 0.02**2, Y > -1.5)] = -1
        d = skfmm.distance(phi, dx)

        fig = plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.imshow(d, vmin=d.min(), vmax=d.max(), origin='lower',
                extent=[X.min(), X.max(), Y.min(), Y.max()])

        fig, ax = plt.subplots()
        plt.gca().set_aspect('equal', adjustable='box')
        CS = ax.contour(X, Y, d,15)
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title('SDF value')
        plt.savefig('sdf_value.png')   

        for i in range(data_size):
            sdf_array.append(np.array(d))

        sdf_array = np.asarray(sdf_array).astype('float32')     

        x_train_1 = np.asarray(sdf_array).astype('float32')
        x_train_2 = np.asarray(x_train_2).astype('float32')

        x_train_1 =np.expand_dims(x_train_1,axis=3)
        x_train_2 =np.expand_dims(x_train_2,axis=2)     

        print('x1 shape',x_train_1.shape)
        print('x2 shape',x_train_2.shape)
        print('y2 shape',y_train_2.shape,y_train_2.max())
        print('y3 shape',y_train_3.shape,y_train_3.max())
        print('y4 shape',y_train_4.shape,y_train_4.max())


#===============================================================
    
    csv_logger=CSVLogger('training_%d.csv'%(exp_no))
    
    checkpoint = ModelCheckpoint('weights.best.hdf5', monitor=user_moni, verbose=1, save_best_only=True, mode='min')    
    print("user_input",user_inp,user_input_channels)
    
    if user_inp=='Y':
            earlystopping=tf.keras.callbacks.EarlyStopping(monitor=user_moni,min_delta=inp_delta,patience=inp_mineps)
            if user_input_channels==3:
                history = model.fit([x_train_1,x_train_2],[y_train_1],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,earlystopping,checkpoint])
            else:
                history = model.fit([x_train_1,x_train_2],[y_train_2,y_train_3,y_train_4],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,earlystopping,checkpoint])
    else:
            if user_input_channels==3:
                history = model.fit([x_train_1,x_train_2],[y_train_1],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,checkpoint])
            else:
                history = model.fit([x_train_1,x_train_2],[y_train_2,y_train_3,y_train_4],batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,checkpoint])    
    
    model.save('Network_Expt_%d.h5'%(exp_no))
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig('h2jet_loss.png')  
    return model
    
        
#def model_prediction(exp_no,model,x1,x2):
def model_prediction(exp_no,model):
    
    # x_test_1
    dx = 0.49/300 
    nx   = 150
    ny   = 150                                                                   
    X, Y = np.meshgrid(np.linspace(-0.15,0.34,150), np.linspace(-0.23,0.26,150))
    phi = 1 * np.ones_like(X)
    phi[np.logical_and(X**2 + Y**2 < 0.02**2, Y > -0.5)] = -1
    d = skfmm.distance(phi, dx)
    sdf_array = []

    test_size = int(200)

    for i in range(test_size):
        sdf_array.append(np.array(d))

    x_test_1 = np.asarray(sdf_array).astype('float32')
    
    
    # x_test_2 : 
    x_test_2 = []
    data = read_csv("./DATA_TRAINING/label_test.csv")      
    test_size=int(test_size)

    Re = data['Reynolds'].tolist()
    Time = data['Time'].tolist()

    x_test_2 = np.ones((test_size,2))
    x_test_2[:,0] = Re
    x_test_2[:,1] = Time
    x_test_2 = np.asarray(x_test_2).astype('float32')
    x_test_2[:,0] = x_test_2[:,0] #/100
    x_test_2[:,1] = x_test_2[:,1] #/273
    print('x1 shape:',x_test_1.shape)
    print('x2 shape:',x_test_2.shape)

    #-.-.-.-.-.-.-.-.-.-

    Predict_list = ['175000']


    # predict
    ans=model.predict([x_test_1,x_test_2])
    

    # khởi tạo để xuất hình 
    p  = []
    u  = []
    v  = []

    P_array = []
    U_array = []
    V_array = []

    # max min of cfd
    P_min, P_max, U_min, U_max, V_min, V_max = find_max_min(Predict_list[0])
    Vmag_min = 0
    Vmag_max = (U_max**2 + V_max**2)**(1/2)
    print('max min of cfd ')
    print(P_min, P_max, U_min, U_max, V_min, V_max, Vmag_min, Vmag_max)

    # max min of predict
    p_min_array  = []
    u_min_array  = []
    v_min_array  = []

    p_max_array  = []
    u_max_array  = []
    v_max_array  = []

    for i in range(test_size):
      p = np.squeeze(ans[0][i], axis=2)
      u = np.squeeze(ans[1][i], axis=2)
      u = u*100
      v = np.squeeze(ans[2][i], axis=2)
      v = v*100
  
      p_min = np.min(p)
      u_min = np.min(u)
      v_min = np.min(v)

      p_max = np.max(p)
      u_max = np.max(u)
      v_max = np.max(v)
      
      p_min_array.append(p_min)
      u_min_array.append(u_min)
      v_min_array.append(v_min)

      p_max_array.append(p_max)
      u_max_array.append(u_max)
      v_max_array.append(v_max)

    p_min_array = np.asarray(p_min_array)
    u_min_array = np.asarray(u_min_array)
    v_min_array = np.asarray(v_min_array)

    p_max_array = np.asarray(p_max_array)
    u_max_array = np.asarray(u_max_array)
    v_max_array = np.asarray(v_max_array)


    p_min = np.min(p_min_array)
    u_min = np.min(u_min_array)
    v_min = np.min(v_min_array)

    p_max = np.max(p_max_array)
    u_max = np.max(u_max_array)
    v_max = np.max(v_max_array)

    vmag_min = 0
    vmag_max = (v_max**2 + u_max**2)**(1/2)


    # xuất ảnh và gif 
    os.makedirs('./PIC')
    for i in range(test_size):
    
      p = np.squeeze(ans[0][i], axis=2)
          
      u = np.squeeze(ans[1][i], axis=2)
      u = u*100

      v = np.squeeze(ans[2][i], axis=2)
      v = v*100
        
            
      # lấy data cfd
      
      data_file = './DATA_TRAINING/'+ Predict_list[0] + '/' + Predict_list[0] + '_' + str(f"{i+1:04n}") +'.txt'
      pic_save_to  = './PIC/'
      
      X_array,Y_array,P_array,U_array,V_array = load_data(data_file)  

      P_array = np.asarray(P_array)
      U_array = np.asarray(U_array*100)
      V_array = np.asarray(V_array*100)

      # tính V magnitude
      vmag = (np.power(u,2) + np.power(v,2))**(1/2)
      Vmag = (np.power(U_array,2) + np.power(V_array,2))**(1/2)
      
      # plot image
      print('plotting image ', data_file)
      plot_image(pic_save_to, P_array, P_min, P_max, 'CFD_','Pressure_'+str(f"{i+1:04n}"),1)            
      plot_image(pic_save_to, U_array, U_min, U_max, 'CFD_','Velocity_'+str(f"{i+1:04n}"),2)
      plot_image(pic_save_to, V_array, V_min, V_max, 'CFD_','Y_Velocity_'+str(f"{i+1:04n}"),3)
      plot_image(pic_save_to, Vmag,Vmag_min, Vmag_max, 'CFD_','V_magnitude_'+str(f"{i+1:04n}"),4)
    

      plot_image(pic_save_to, p, p_min, p_max, 'Predict_','Pressure_'+str(f"{i+1:04n}"),1)            
      plot_image(pic_save_to, u, u_min, u_max, 'Predict_','Velocity_'+str(f"{i+1:04n}"),2)
      plot_image(pic_save_to, v, v_min, v_max, 'Predict_','Y_Velocity_'+str(f"{i+1:04n}"),3)
      plot_image(pic_save_to, vmag, vmag_min, vmag_max, 'Predict_','V_magnitude_'+str(f"{i+1:04n}"),4)
      

    # plot gif
    print('Plotting gif...')
    plot_gif('CFD_Pressure')
    plot_gif('CFD_Velocity')
    plot_gif('CFD_Y_Velocity')
    plot_gif('CFD_V_magnitude')

    plot_gif('Predict_Pressure')
    plot_gif('Predict_Velocity')
    plot_gif('Predict_Y_Velocity')
    plot_gif('Predict_V_magnitude')
        
    return ans

def save_model(model,modelname):            
    model.save(modelname+'.h5')
    return


def load(modelname):                        
    model=load_model(modelname)
    return model



def plot_image(folder_out,var,vmin,vmax,pretext,fieldname,flag):

    if (flag==1):
      labeltxt = 'Pressure'

        # P_min  0.7 P_max 1.2
      X, Y = np.meshgrid(np.linspace(-0.15,0.34,150), np.linspace(-0.23,0.26,150))        
    
      fig, ax = plt.subplots()
                                   
      plt.gca().set_aspect('equal', adjustable='box')     #
      levels = np.linspace(vmin,vmax,50)
      plt.contourf(X, Y, var, levels,cmap=plt.cm.rainbow,origin='lower',
                 extent=[X.min(), X.max(), Y.min(), Y.max()])     # 
      plt.colorbar(label=labeltxt)                        
      plt.savefig(folder_out+pretext+fieldname+'.png')               
      plt.close()

    elif (flag==2):                                  
      labeltxt = 'X Velocity (m/s)'

        # umin = -80 , max 230

      X, Y = np.meshgrid(np.linspace(-0.15,0.34,150), np.linspace(-0.23,0.26,150))        
    
      fig, ax = plt.subplots()
                                   
      plt.gca().set_aspect('equal', adjustable='box')     #
      levels = np.linspace(vmin,vmax,50)
      plt.contourf(X, Y, var, levels,cmap=plt.cm.rainbow,origin='lower',
                 extent=[X.min(), X.max(), Y.min(), Y.max()])     # 
      plt.colorbar(label=labeltxt)                        
      plt.savefig(folder_out+pretext+fieldname+'.png')               
      plt.close()

    elif (flag==3):                                  
      labeltxt = 'Y Velocity (m/s)'

        # vmin = -130, max = 130
      X, Y = np.meshgrid(np.linspace(-0.15,0.34,150), np.linspace(-0.23,0.26,150))        
    
      fig, ax = plt.subplots()
                                   
      plt.gca().set_aspect('equal', adjustable='box')     #
      levels = np.linspace(vmin,vmax,50)
      plt.contourf(X, Y, var, levels,cmap=plt.cm.rainbow,origin='lower',
                 extent=[X.min(), X.max(), Y.min(), Y.max()])     # 
      plt.colorbar(label=labeltxt)                        
      plt.savefig(folder_out+pretext+fieldname+'.png')               
      plt.close()
    
    elif (flag==4):                                  
      labeltxt = 'V magnitude (m/s)'

        # vmin = -130, max = 130
      X, Y = np.meshgrid(np.linspace(-0.15,0.34,150), np.linspace(-0.23,0.26,150))        
    
      fig, ax = plt.subplots()
                                   
      plt.gca().set_aspect('equal', adjustable='box')     #
      levels = np.linspace(vmin,vmax,50)
      plt.contourf(X, Y, var, levels,cmap=plt.cm.rainbow,origin='lower',
                 extent=[X.min(), X.max(), Y.min(), Y.max()])     # 
      plt.colorbar(label=labeltxt)                        
      plt.savefig(folder_out+pretext+fieldname+'.png')               
      plt.close()


def plot_gif(name):

  fp_in = './PIC/' + name + '*.png'
  fp_out = name + '.gif'

    # gif chạy 10s
  img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
  img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=50, loop=0)
  
def find_max_min(Re):

  P_min_array  = []
  U_min_array  = []
  V_min_array  = []

  P_max_array  = []
  U_max_array  = []
  V_max_array  = []

  for i in range(0,200):  
    data_file = './DATA_TRAINING/'+ Re + '/' + Re + '_' + str(f"{i+1:04n}")+'.txt'   
    print('finding max min of ...',data_file)
    X_array,Y_array,P_array,U_array,V_array = load_data(data_file)

    P_array = np.asarray(P_array)
    U_array = np.asarray(U_array*100)
    V_array = np.asarray(V_array*100)
  
    P_min = np.min(P_array)
    U_min = np.min(U_array)
    V_min = np.min(V_array)

    P_max = np.max(P_array)
    U_max = np.max(U_array)
    V_max = np.max(V_array)
    
    P_min_array.append(P_min)
    U_min_array.append(U_min)
    V_min_array.append(V_min)

    P_max_array.append(P_max)
    U_max_array.append(U_max)
    V_max_array.append(V_max)

  P_min_array = np.asarray(P_min_array)
  U_min_array = np.asarray(U_min_array)
  V_min_array = np.asarray(V_min_array)

  P_max_array = np.asarray(P_max_array)
  U_max_array = np.asarray(U_max_array)
  V_max_array = np.asarray(V_max_array)


  P_min = np.min(P_min_array)
  U_min = np.min(U_min_array)
  V_min = np.min(V_min_array)

  P_max = np.max(P_max_array)
  U_max = np.max(U_max_array)
  V_max = np.max(V_max_array)

  print(P_min, P_max, U_min, U_max, V_min, V_max)


  return P_min, P_max, U_min, U_max, V_min, V_max

