import os
import sys

import matplotlib
matplotlib.use( 'Agg' )
from matplotlib import pyplot


def plotHistory( history ):
	
    """
    Placeholder
    """

    # plot loss
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='red', label='test')
    
    # dictionary key names recently updated ( keras 2.2.5 -> 2.3.0 )
    keys = [ 'acc', 'accuracy' ] 
    for k in keys:
        if k in history.history:

            # plot train and validation accuracies
            pyplot.subplot(212)
            pyplot.title('Classification Accuracy')
            pyplot.plot(history.history[ k ], color='blue', label='train')
            pyplot.plot(history.history['val_{}'.format( k ) ], color='red', label='test')
            break
    
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

    return
