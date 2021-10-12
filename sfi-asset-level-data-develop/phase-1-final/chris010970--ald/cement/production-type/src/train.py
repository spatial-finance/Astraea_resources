import os
import importlib

# keras
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.layers.core import Dropout
from keras.models import clone_model

# tensorflow
import tensorflow as tf 
from tensorflow.keras.metrics import AUC, Precision, Recall

# utility
from utility.dl.diagnostics import plotHistory
from utility.dl.args import parseTrainArguments
from utility.dl.cnn import getVgg16, getResNet50, getInceptionV3
from utility.dl.cnn import loadFromFile, saveToFile
from utility.dl.scale import getBinaryClassWeights
from utility.dl.losses import binary_focal_loss

def main():

    """ 
    setup model based on command line input and execute training 
    """

    # parse arguments
    args = parseTrainArguments()
    print ( 'epochs {} / batch size {}'.format ( args.epochs, args.batch_size ) )

    # start session
    session = tf.keras.backend.get_session()
    init = tf.global_variables_initializer()
    session.run(init)

    # optional model creation
    if args.load_path is not None:

        # load model from file
        model, args.model = loadFromFile( args.load_path )

    else:

        # define topmost cnn layers plugged into pretrained model
        layers = {  'fc' : [    { 'units' : 256, 'activation' : 'tanh', 'dropout' : 0.2 },
                                { 'units' : 128, 'activation' : 'tanh', 'dropout' : 0.2 } ],
                    'out' : [   { 'units' : 1, 'activation' : 'sigmoid' } ]
        }

        # create model from argument
        cnn_library = { 'vgg16': getVgg16, 'resnet50': getResNet50, 'inception_v3': getInceptionV3 }
        model = cnn_library[ args.model ]( ( args.image_size, args.image_size, 3 ), layers )


    # valid model
    plot_model(model, show_shapes=True, to_file='model.png')
    if model is not None: 

        # setup optimiser and compile
        #opt = SGD( lr=0.01, momentum=0.9 )
        opt = Adam( lr=1e-6 )
        model.compile(  optimizer=opt, 
                        loss='binary_crossentropy', 
                        metrics=[AUC(name='auc'),Precision(name='precision'),Recall(name='recall') ] )
        
        #opt = Adam( lr=1e-6 )
        #model.compile(  optimizer=opt, 
        #                loss=[binary_focal_loss(alpha=.90, gamma=2)], 
        #                metrics=[AUC(name='auc'),Precision(name='precision'),Recall(name='recall') ] )
        model.summary()

        # select preprocess_input wrapper
        module = importlib.import_module( 'keras.applications.{}'.format( args.model ) )
        preprocess_input = module.preprocess_input
        
        # create data generators
        train_datagen = ImageDataGenerator( preprocessing_function=preprocess_input,
                                            horizontal_flip=True, 
                                            vertical_flip=True, 
                                            rotation_range=90 )
        
        test_datagen = ImageDataGenerator(  preprocessing_function=preprocess_input )

        # get train iterator - binary classification
        path = os.path.join( args.data_path, 'train' )
        train_it = train_datagen.flow_from_directory(   path, 
                                                        class_mode='binary', 
                                                        batch_size=args.batch_size, 
                                                        classes=[ 'dry', 'wet' ],
                                                        target_size=(args.image_size, args.image_size) )

        # get test iterator - binary classification
        path = os.path.join( args.data_path, 'test' )
        test_it = test_datagen.flow_from_directory( path, 
                                                    class_mode='binary', 
                                                    batch_size=args.batch_size, 
                                                    classes=[ 'dry', 'wet' ],
                                                    target_size=(args.image_size, args.image_size) )

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
            path = os.path.join( args.checkpoint_path, "weights-{epoch:02d}-{val_accuracy:.2f}.h5" )
            checkpoint = ModelCheckpoint(   path, 
                                            monitor='val_accuracy', 
                                            verbose=1, 
                                            save_best_only=True, 
                                            mode='max' )
            callbacks.append( checkpoint )


        # fit model
        weights = getBinaryClassWeights( args.data_path, [ 'dry', 'wet' ] )
        history = model.fit_generator(  train_it, 
                                        steps_per_epoch=len(train_it), 
                                        class_weight=weights,
                                        validation_data=test_it, 
                                        validation_steps=len(test_it), 
                                        epochs=args.epochs, 
                                        callbacks=callbacks,
                                        verbose=1 )

        # evaluate model
        scores = model.evaluate_generator(  test_it, 
                                            steps=len(test_it), 
                                            verbose=1 )
        print('Final Metric Scores> {}'.format( scores ) )

        # optional save
        if args.save_path is not None:
            saveToFile( model, args.save_path, args.model )

        # plot learning curves
        plotHistory(history)

    return
    
# execute main
if __name__ == '__main__':
    main()
