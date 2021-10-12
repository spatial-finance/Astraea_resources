import os
import sys
import importlib
import pandas as pd

# keras
import tensorflow as tf
from keras.utils import plot_model
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# utility
from utility.dl.args import parseTrainArguments
from utility.dl.diagnostics import plotHistory
from utility.dl.cnn import saveToFile, loadFromFile
from utility.dl.cnn import getVgg16, getResNet50, getInceptionV3


def main():

    """ 
    setup model based on command line input and execute training 
    """

    # parse arguments
    args = parseTrainArguments()
    print ( 'epochs {} / batch size {}'.format ( args.epochs, args.batch_size ) )

    # create tf session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # optional model creation
    if args.load_path is not None:

        # load model from file
        model, args.model = loadFromFile( args.load_path )

    else:

        # define topmost cnn layers plugged into pretrained model
        layers = {  'fc' : [    { 'units' : 256, 'activation' : 'relu', 'dropout' : 0.2 },
                                { 'units' : 128, 'activation' : 'relu', 'dropout' : 0.2 } ],
                    'out' : [   { 'units' : 1, 'activation' : 'linear' } ]
        }

        # create model from argument
        cnn_library = { 'vgg16': getVgg16, 'resnet50': getResNet50, 'inception_v3' : getInceptionV3 }
        model = cnn_library[ args.model ]( ( args.image_size, args.image_size, 3 ), layers )

    # valid model
    # plot_model(model, show_shapes=True, to_file='model.png')
    if model is not None: 

        # setup optimiser and compile
        opt = Adam(lr=1e-6)
        model.compile(loss='mean_absolute_error', optimizer=opt)

        # select preprocess_input wrapper
        module = importlib.import_module( 'keras.applications.{}'.format( args.model ) )
        preprocess_input = module.preprocess_input

        # create data generators
        train_datagen = ImageDataGenerator( preprocessing_function=preprocess_input,
                                            horizontal_flip=True, 
                                            vertical_flip=True, 
                                            rotation_range=90 )

        # fit the data augmentation
        test_datagen = ImageDataGenerator( preprocessing_function=preprocess_input )
        scaler = MinMaxScaler()
    
        # load train dataframe
        df_train = pd.read_csv( os.path.join( args.data_path, 'train.csv' ) )
        df_train[ 'target' ] = scaler.fit_transform( df_train[ [ 'target'] ] )
        
        # create train iterator        
        data_path = os.path.join( args.data_path, 'train' )
        train_it = train_datagen.flow_from_dataframe(   dataframe=df_train,
                                                        directory=data_path,
                                                        x_col='image',
                                                        y_col='target',
                                                        class_mode='raw',
                                                        color_mode='rgb',
                                                        shuffle=True,
                                                        target_size=(args.image_size, args.image_size),
                                                        batch_size=args.batch_size )

        # create test iterator
        df_test = pd.read_csv( os.path.join( args.data_path, 'test.csv' ) )
        df_test[ 'target' ] = scaler.transform( df_test[ [ 'target'] ] )

        # data_path = os.path.join( os.path.join( args.data_path, 'test' ), name )
        data_path = os.path.join( args.data_path, 'test' )
        test_it = test_datagen.flow_from_dataframe(     dataframe=df_test,
                                                        directory=data_path,
                                                        x_col='image',
                                                        y_col='target',
                                                        class_mode='raw',
                                                        color_mode='rgb',
                                                        shuffle=True,
                                                        target_size=(args.image_size, args.image_size),
                                                        batch_size=args.batch_size )

        # confirm the iterator works
        batchX, batchy = train_it.next()
        print('Batch shape=%s, min=%.3f, max=%.3f, mean=%.3f, std=%.3f' % (batchX.shape, batchX.min(), batchX.max(), batchX.mean(), batchX.std() ))

        # setup callbacks
        callbacks = [ CSVLogger( 'log.csv', append=True ) ]
        if args.checkpoint_path is not None:

            # create sub-directory if required
            if not os.path.exists ( args.checkpoint_path ):
                os.makedirs( args.checkpoint_path )

            # setup checkpointing callback
            path = os.path.join( args.checkpoint_path, "weights-{epoch:02d}-{val_loss:.2f}.h5" )
            checkpoint = ModelCheckpoint(   path, 
                                            monitor='val_loss', 
                                            verbose=1, 
                                            save_best_only=True, 
                                            mode='min' )
            callbacks.append( checkpoint )

        # execute fit
        history = model.fit_generator(  train_it, 
                                        steps_per_epoch=len(train_it), 
                                        validation_data=test_it, 
                                        validation_steps=len(test_it), 
                                        epochs=args.epochs, 
                                        callbacks=callbacks,
                                        verbose=1 )

        # optional save
        if args.save_path is not None:
            saveToFile( model, args.save_path, args.model )

        # plot learning curves
        plotHistory(history)

    return

    
# execute main
if __name__ == '__main__':
    main()
