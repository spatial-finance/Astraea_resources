import os
import importlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn + keras
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import ImageDataGenerator

# add utility functions
from utility.dp import getUniqueId
from utility.dl.cnn import loadFromFile


def getPrediction( datagen, model, df, data_path ):

	"""
	get regression
	"""

	# create iterator        
	it = datagen.flow_from_dataframe(   dataframe=df,
										directory=data_path,
										x_col='image',
										y_col='target',
										class_mode='raw',
										color_mode='rgb',
										shuffle=False,
										target_size=(256,256),
										batch_size=1 )

	# run prediction
	df[ 'yhat' ] = model.predict_generator( it )
	return df


def plotRegression( dfs ):

	"""
	plot regression
	"""

	# create figure
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
	for idx, df in enumerate( dfs ):

		# compute regression
		m, c, r2, p, err = stats.linregress( df['target'].values, df['yhat'].values )    

		# plot sample data and regression model
		axes[ idx ].plot( df['target'].values, df['yhat'].values, '.' )
		axes[ idx ].plot( [0, 1], [c, m+c], '-', label='y={:.2f}x+{:.2f}\nR2={:.2f}'.format( m, c, r2 ) )
		axes[ idx ].plot( [0, 1], [0, 1], '--', color='g', label='1-to-1' )

		# fix axes and plot 1-2-1 line
		axes[ idx ].set_xlim([0,1])
		axes[ idx ].set_ylim([0,1])

		subset = 'Train' if idx == 0 else 'Test'
		axes[ idx ].set_title( 'Actual vs Estimated Capacity: {}'.format( subset ) )
		axes[ idx ].legend( fontsize=9 )

	plt.show()
	return


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

	# plot subset sample sizes for wet and dry classes
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

	# get file list length - sample size
	count = [None] * 2
	for idx, s in enumerate( [ 'train', 'test' ] ):
		path = os.path.join( data_path, s )
		count[ idx ] = len( os.listdir( path ) )
			
	# draw counts as subplot
	axes.set_title( 'Sample sizes' )
	axes.bar( [ 'train', 'test' ], count )

	plt.show()
	return


def parseArguments(args=None):

    """
    parse command line argument
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

	# load pre-trained model from file 
	model, model_type = loadFromFile( args.model_path )

	# select preprocess_input wrapper
	module = importlib.import_module( 'keras.applications.{}'.format( model_type ) )
	preprocess_input = module.preprocess_input
			
	datagen = ImageDataGenerator(  preprocessing_function=preprocess_input )
	scaler = MinMaxScaler()

	# plot sample size plots and loss diagnostics
	plotSampleSizes( args.data_path )
	plotDiagnostics( args.model_path )

	# read dataframe and normalise target
	df_train = pd.read_csv( os.path.join( args.data_path, 'train.csv' ) )
	df_train[ 'target' ] = scaler.fit_transform( df_train[ [ 'target'] ] )

	#df_train = getPrediction( datagen, model, df_train, os.path.join( args.data_path, 'train' ) )

	# read dataframe and normalise target
	df_test = pd.read_csv( os.path.join( args.data_path, 'test.csv' ) )
	df_test[ 'target' ] = scaler.transform( df_test[ [ 'target'] ] )

	#df_test = getPrediction( datagen, model, df_test, os.path.join( args.data_path, 'test' ) )

	# plot regression
	#plotRegression( [ df_train, df_test ] )

	# finally run model against unlabelled images - unknown capacity
	path = os.path.join( args.data_path, 'unlabelled' )
	it = datagen.flow_from_directory( path, 
									classes=['test'], 
									color_mode='rgb',
									shuffle=False,
									batch_size=1, 
									target_size=(args.image_size, args.image_size) )

	# evaluate probabilities
	y_pred = model.predict_generator( it )

	# compile results
	records = []
	for idx, filename in enumerate ( it.filenames ):	

		# assign label and confidence
		records.append (  { 	'uid':  getUniqueId( filename ),		
								'capacity' : float( np.exp( y_pred[ idx ] )  ) } )

	# convert to dataframe
	df = pd.DataFrame.from_dict( records )

	# compute mean for each uid - drop duplicates
	df[ 'mean' ] = df.groupby(['uid']).capacity.transform( 'mean' )
	df.drop_duplicates(subset='uid', keep='first', inplace=True ) 
	df = df.drop( columns=[ 'capacity' ] )

	for idx, row in df.iterrows():
		print ( row[ 'uid' ], row[ 'mean' ] )

	# create figure
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(14, 6))
	axes.set_title( 'Model Predicted Capacity for Unlabelled Cement Factory Sites' )
	axes.set_ylabel('Mt / year') 

	axes.set_xticks(range(0, len(df) ) )
	axes.tick_params(axis='both', which='major', labelsize=8)

	axes.set_xticklabels( df[ 'uid' ].tolist(), rotation=90 )
	axes.plot( df[ 'mean' ].tolist() )

	# show figure
	fig.tight_layout(rect=[0, 0.05, 1, 0.95])
	plt.show()

	return


# execute main
if __name__ == '__main__':
    main()
