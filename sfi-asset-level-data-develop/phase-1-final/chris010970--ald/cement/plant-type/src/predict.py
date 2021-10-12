import os
import importlib
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# keras
from keras.preprocessing.image import ImageDataGenerator

# add utility functions
from utility.dl.cnn import loadFromFile
from sklearn.metrics import confusion_matrix


def getWronguns( it, y_pred ):

	"""
	get list of wrongly classified images
	"""

	# get pathnames of incorrectly classified images
	wronguns = { 'grinding': [], 'integrated' : [] }
	for idx, label in enumerate( it.class_indices.keys() ):
	
		items = np.argwhere( (it.classes==idx) & (y_pred.flatten()==1-idx) ).flatten()
		wronguns[ label ] = [ it.filepaths[i] for i in items ]

	return wronguns


def plotConfusionMatrix( cms ):

	"""
	plot train and test confusion matrix
	"""

	# create figure
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
	for idx, cm in enumerate( cms ):

		# font and label size
		sn.set(font_scale=1.1) 
		sn.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='.2f', ax=axes[ idx ] )

		subset = 'Train' if idx == 0 else 'Test'
		axes[ idx ].set_title( 'Normalised Confusion Matrix: {}'.format( subset ) )

	plt.show()
	return


def getConfusionMatrix( it, y_pred ):

	"""
	compute confusion matrix for prediction
	"""

	# compute normalised confusion matrix 
	cm = confusion_matrix( it.classes, y_pred )
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	# parse normalised confusion matrix into dataframe
	classes = list( it.class_indices.keys() )
	return pd.DataFrame( cm, index=classes, columns=classes )


def getPrediction( model, datagen, args, subset ):

	"""
	generate prediction for images in path directory
	"""

	# get test iterator - binary classification
	path = os.path.join( args.data_path, subset )
	it = datagen.flow_from_directory( path, 
									class_mode='binary', 
									color_mode='rgb',
									shuffle=False,
									batch_size=1, 
									target_size=(args.image_size, args.image_size) )

	# run prediction
	probabilities = model.predict_generator( it )
	return it, probabilities > 0.5


def getResults( model, datagen, args, subset ):

	"""
	generate results in form of a confusion matrix
	"""

	# get prediction
	it, y_pred = getPrediction( model, datagen, args, subset )

	# compute confusion matrix
	cm = getConfusionMatrix( it, y_pred )

	# compile list of wrongly classified images
	wronguns = getWronguns( it, y_pred )

	return cm, wronguns


def plotDiagnostics( model_path ):

	"""
	plot diagnostics
	"""

	# plot diagnostic information
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
	df = pd.read_csv( os.path.join( model_path, 'log.csv' ) )
		
	# plot diagnostics of model 
	axes.plot( df['loss'].values, '-', label='train' )
	
	axes.set_title( 'Loss Function: Mean Absolute Error' )
	axes.legend( fontsize=9 )	

	plt.show()
	return


def plotSampleSizes( data_path ):

	"""
	plot sample sizes
	"""

	# get file list length - sample size
	count = {}
	for s in [ 'train', 'test' ]:

		path = os.path.join( data_path, s )
		count[ s ] = [None] * 2

		for idx, c in enumerate( [ 'grinding', 'integrated' ] ):
			count[ s ] [ idx ] = len( os.listdir( os.path.join( path, c ) ) )

	# create figure
	fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 4))
	for idx, s in enumerate( [ 'train', 'test' ] ):

		axes[ idx ].barh( [ 'grinding', 'integrated'  ], count[ s ] )
		axes[ idx ].set_title( 'Sample Size: {}'.format( s ) )

	# show figure
	fig.tight_layout(rect=[0, 0.05, 1, 0.95])
	plt.show()
	return


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description='data prep')
    parser.add_argument('model_path', action="store")
    parser.add_argument('data_path', action="store")

    return parser.parse_args(args)


def main():

	"""
	main path of execution
	"""

	# parse arguments
	args = parseArguments()
	args.image_size = 256

	# load model
	model, args.model = loadFromFile( args.model_path )
	plotSampleSizes( args.data_path )

	# select preprocess_input wrapper
	module = importlib.import_module( 'keras.applications.{}'.format( args.model ) )
	preprocess_input = module.preprocess_input
        
	# create test generator and compute results
	datagen = ImageDataGenerator( preprocessing_function=preprocess_input )

	train_cm, train_wronguns = getResults( model, datagen, args, 'train' )
	test_cm, test_wronguns = getResults( model, datagen, args, 'test' )

	# plot confusion matrices
	plotConfusionMatrix( [ train_cm, test_cm ] )

	return


# execute main
if __name__ == '__main__':
    main()
