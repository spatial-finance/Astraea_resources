import os
import glob

from PIL import Image
from numpy import asarray


def getGlobalStandardizeImage( pathname ):

    """
    Placeholder
    """

    # load image
    img = Image.open( pathname )
    data = asarray( img )

    # convert from integers to floats
    data = data.astype('float32')

    # calculate global mean and standard deviation
    mean, std = data.mean(), data.std()
    # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std) )

    # global standardization of pixels
    data = (data - mean) / std

    # confirm it had the desired effect
    mean, std = data.mean(), data.std()
    # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std) )

    return data


def getLocalStandardizeImage( pathname, target_size=None ):

    """
    Placeholder
    """

    # load and maybe resize image
    img = Image.open( pathname )
    if target_size is not None:
        img = img.resize( target_size )

    # convert from integers to floats
    data = asarray(img)
    data = data.astype('float32')

    # calculate per-channel means and standard deviations
    means = data.mean(axis=(0,1), dtype='float64')
    stds = data.std(axis=(0,1), dtype='float64')
    #print('Means: %s, Stds: %s' % (means, stds))

    # per-channel standardization of pixels
    data = ( data - means) / stds

    # confirm it had the desired effect
    means = data.mean(axis=(0,1), dtype='float64')
    stds = data.std(axis=(0,1), dtype='float64')
    #print('Means: %s, Stds: %s' % (means, stds))

    return data 


def getBinaryClassWeights( root_path, classes ):

    """
    Placeholder
    """

    # construct subset path for each class
    samples = dict.fromkeys( classes, 0 )
    for c in classes:

        subsets = [ 'train', 'validation', 'test' ]
        for s in subsets:

            # increment sample count for class
            path = os.path.join( os.path.join( root_path, s ), c )
            samples[ c ] += len( glob.glob( path + '/*.jpg' ) )

    # get total samples
    total_samples = 0
    for c in classes:
        total_samples += samples[ c ]

    # compute weightings
    weights = dict.fromkeys( range( 0, len( classes ) ) )
    for idx, c in enumerate( classes ):
        weights[ idx ] = total_samples / ( 2.0 * samples[ c ] )
    
    return weights

