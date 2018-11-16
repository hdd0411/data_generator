###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD

import sys
sys.path.insert(0, './lib/')
from help_functions import *
from keras.layers import BatchNormalization
#function to obtain data for training/testing (validation)
from extract_patches import get_data_training
from keras.layers.core import  Dropout, Activation
from keras import backend as K
print(K.backend())
from data_feed import *
from pre_processing import my_PreProc
import math
#Define the neural network
def block_2_conv(input,num_filter):
    conv1=Conv2D(num_filter,(3,3),strides=(1,1),padding='same',data_format='channels_first')(input)
    conv1_bn=BatchNormalization(axis=1)(conv1)
    conv1_relu=Activation('relu')(conv1_bn)
    conv2=Conv2D(num_filter,(3,3),strides=(1,1),padding='same',data_format='channels_first')(conv1_relu)
    conv2_bn=BatchNormalization(axis=1)(conv2)
    conv2_add=Add()([input,conv2_bn])
    conv2_relu=Activation('relu')(conv2_add)
    return conv2_relu
# F1 score: harmonic mean of precision and sensitivity DICE = 2*TP/(2*TP + FN + FP)
def DiceCoef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f*y_pred_f)
	return (2.*intersection)/(K.sum(y_true_f) + K.sum(y_pred_f) + 0.00001)

def DiceCoefLoss(y_true, y_pred):
	return -DiceCoef(y_true, y_pred)

def get_unet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv5)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    ############
    conv7 = core.Activation('softmax')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    sgd=SGD(lr=0.01, decay=5e-4, momentum=0.99)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    return model
def get_unet_new(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv0=Conv2D(8,(1,1),padding='same')(inputs)
    conv1=block_2_conv(conv0,8)
    conv1_1=Conv2D(16,(2,2),strides=2,data_format='channels_first')(conv1)
    conv1_1=BatchNormalization(axis=1)(conv1_1)
    conv1_1=Activation('relu')(conv1_1)

    conv2=block_2_conv(conv1_1,16)
    conv2_1=Conv2D(32,(2,2),strides=2,data_format='channels_first')(conv2)
    conv2_1=BatchNormalization(axis=1)(conv2_1)
    conv2_1=Activation('relu')(conv2_1)

    conv3=block_2_conv(conv2_1,32)
    conv3_1=Conv2D(64,(2,2),strides=2,data_format='channels_first')(conv3)
    conv3_1=BatchNormalization(axis=1)(conv3_1)
    conv3_1=Activation('relu')(conv3_1)

    conv4=block_2_conv(conv3_1,64)
    conv4_1=Conv2D(128,(2,2),strides=2,data_format='channels_first')(conv4)
    conv4_1=BatchNormalization(axis=1)(conv4_1)
    conv4_1=Activation('relu')(conv4_1)

    conv5=block_2_conv(conv4_1,128)
    conv5_dropout=Dropout(0.5)(conv5)
    conv5_1=Conv2D(256,(2,2),strides=2,data_format='channels_first')(conv5_dropout)
    conv5_1=BatchNormalization(axis=1)(conv5_1)
    conv5_1=Activation('relu')(conv5_1)

    conv6=block_2_conv(conv5_1,256)
    conv6_dropout=Dropout(0.5)(conv6)
    conv6_1=Conv2D(512,(2,2),strides=2,data_format='channels_first')(conv6_dropout)
    conv6_1=BatchNormalization(axis=1)(conv6_1)
    conv6_1=Activation('relu')(conv6_1)

    conv7=block_2_conv(conv6_1,512)
    print (conv7.shape)
    up1=UpSampling2D(size=(2, 2))(conv7)
    up1_1= Conv2D(256,(2,2),strides=1,padding='same',data_format='channels_first')(up1)
    up1_1=BatchNormalization(axis=1)(up1_1)
    up1_1=Activation('relu')(up1_1)
    up1_2=concatenate([conv6,up1_1],axis=1)
    up1_3=block_2_conv(up1_2,512)

    up2=UpSampling2D(size=(2,2))(up1_3)
    up2_1=Conv2D(128,(2,2),strides=1,padding='same',data_format='channels_first')(up2)
    up2_1=BatchNormalization(axis=1)(up2_1)
    up2_1=Activation('relu')(up2_1)
    up2_2=concatenate([conv5,up2_1],axis=1)
    up2_3=block_2_conv(up2_2,256)

    up3=UpSampling2D(size=(2,2))(up2_3)
    up3_1=Conv2D(64,(2,2),strides=1,padding='same',data_format='channels_first')(up3)
    up3_1=BatchNormalization(axis=1)(up3_1)
    up3_1=Activation('relu')(up3_1)
    up3_2=concatenate([conv4,up3_1],axis=1)
    up3_3=block_2_conv(up3_2,128)

    up4=UpSampling2D(size=(2,2))(up3_3)
    up4_1=Conv2D(32,(2,2),strides=1,padding='same',data_format='channels_first')(up4)
    up4_1=BatchNormalization(axis=1)(up4_1)
    up4_1=Activation('relu')(up4_1)
    up4_2=concatenate([conv3,up4_1],axis=1)
    up4_3=block_2_conv(up4_2,64)

    up5=UpSampling2D(size=(2,2))(up4_3)
    up5_1=Conv2D(16,(2,2),strides=1,padding='same',data_format='channels_first')(up5)
    up5_1=BatchNormalization(axis=1)(up5_1)
    up5_1=Activation('relu')(up5_1)
    up5_2=concatenate([conv2,up5_1],axis=1)
    up5_3=block_2_conv(up5_2,32)

    up6=UpSampling2D(size=(2,2))(up5_3)
    up6_1=Conv2D(8,(2,2),strides=1,padding='same',data_format='channels_first')(up6)
    up6_1=BatchNormalization(axis=1)(up6_1)
    up6_1=Activation('relu')(up6_1)
    up6_2=concatenate([conv1,up6_1],axis=1)
    up6_3=block_2_conv(up6_2,16)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up6_3)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='sgd', loss=DiceCoefLoss, metrics=[DiceCoef])
    return model







#Define the neural network gnet
#you need change function call "get_unet" to "get_gnet" in line 166 before use this network
def get_gnet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    up1 = UpSampling2D(size=(2, 2))(conv1)
    #
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)
    #
    up2 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)
    #
    up3 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)
    #
    up4 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up4)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv8)
    #
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    #
    conv10 = Convolution2D(2, 1, 1, activation='relu', border_mode='same')(conv9)
    conv10 = core.Reshape((2,patch_height*patch_width))(conv10)
    conv10 = core.Permute((2,1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

    return model

#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
_batchSize = int(config.get('training settings', 'batch_size'))



#============ Load the data and divided in patches
# patches_imgs_train, patches_masks_train = get_data_training(
#     DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
#     DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
#     patch_height = int(config.get('data attributes', 'patch_height')),
#     patch_width = int(config.get('data attributes', 'patch_width')),
#     N_subimgs = int(config.get('training settings', 'N_subimgs')),
#     inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
# )


#========= Save a sample of what you're feeding to the neural network ==========
# N_sample = min(patches_imgs_train.shape[0],40)
# visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
# visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model arcitecture =====
# n_ch = patches_imgs_train.shape[1]
# patch_height = patches_imgs_train.shape[2]
# patch_width = patches_imgs_train.shape[3]
n_ch=4
patch_height=512
patch_width=512
model = get_unet_new(n_ch, patch_height, patch_width)  #the U-net model
print ("Check: final output of the network:")
print (model.output_shape)
#plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)




new_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original')
new_train_imgs_groundTruth=path_data + config.get('data paths', 'train_groundTruth')
train_data= h5py.File(new_train_imgs_original,'r')
train_imgs_original= np.array(train_data['data'])
train_groundTruth=np.array(train_data['label'])

#train_groundTruth = load_hdf5(new_train_imgs_groundTruth) #masks always the same


train_imgs = train_imgs_original/255.
train_masks = train_groundTruth/255.

#check masks are within 0-1
assert(np.min(train_masks)==0 and np.max(train_masks)==1)
print ("\ntrain images/masks shape:")
print (train_imgs.shape)
print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
print ("train masks are within 0-1\n")
#============  Training ==================================
checkpoint_test = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', monitor='val_loss', save_best_only=True,save_weights_only=True) #save at each epoch if the validation decreased
checkpoint = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment + "bestTrainWeight" + ".h5", monitor='loss', save_best_only=True, save_weights_only=True)

# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

#patches_masks_train = masks_Unet(patches_masks_train)  #reduce memory consumption
#model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer])

keepPctOriginal = 0.5
trans = 0.16 # +/- proportion of the image size
rot = 9 # +/- degree
zoom = 0.12 # +/- factor
shear = 0 # +/- in radian x*np.pi/180
elastix = 0 # in pixel
intensity = 0.07 # +/- factor
hflip = True
vflip = True
iter_times=32
num=train_imgs_original.shape[0]
index=list(np.random.permutation(num))
_X_train=train_imgs[index][0:16]
_Y_train=train_masks[index][0:16]
print(_X_train.shape)
print(_Y_train.shape)
_X_vali=train_imgs[index][16:20]
_Y_vali=train_masks[index][16:20]
print(_X_vali.shape)
print(_Y_vali.shape)


def ImgGenerator():
    for image in train_generator(_X_train, _Y_train,_batchSize, iter_times, _keepPctOriginal=0.5,
                                 _trans=TRANSLATION_FACTOR, _rot=ROTATION_FACTOR, _zoom=ZOOM_FACTOR, _shear=SHEAR_FACTOR,
                                 _elastix=VECTOR_FIELD_SIGMA, _intensity=INTENSITY_FACTOR, _hflip=True, _vflip=True):
          yield image
def valiGenerator():
    for image in validation_generator(_X_vali, _Y_vali,_batchSize):
        yield image

stepsPerEpoch = math.ceil((num-4) / _batchSize)
validationSteps = math.ceil(4 / _batchSize)
history = model.fit_generator(ImgGenerator(), verbose=2, workers=1,
                                                 validation_data = valiGenerator(),
                                                 steps_per_epoch=stepsPerEpoch, epochs=N_epochs,
                                                 validation_steps=validationSteps,
                                                 callbacks=[checkpoint,checkpoint_test])
model.summary()
#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
#test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


















#
