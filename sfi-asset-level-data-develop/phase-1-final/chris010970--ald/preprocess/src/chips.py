import os
import glob
import argparse
import numpy as np

from PIL import Image
from scipy import stats
from utility.dp import getUniqueId
from sklearn.model_selection import train_test_split

# disable decompression bomb exceptions
Image.MAX_IMAGE_PIXELS = None


def getImageChips( image, out_path, crops=[1024, 2048], resize=256 ):

    """
    crop image at various scales and resize to common shape
    """

    # create out path
    if not os.path.exists( out_path ):
        os.makedirs( out_path )

    # open source image
    src = Image.open( image )
    for crop in crops:

        # create unique pathname based on crop and resize
        filename = os.path.basename( image ).replace( '.tif', '_{}_{}.jpg'.format( crop, resize ) ) 
        out_pathname = os.path.join( out_path, filename )

        if not os.path.exists( out_pathname ):

            print ( out_pathname )

            # crop around central pixel location
            centre = tuple([ int( z / 2 ) for z in src.size ] ) 
            crop2 = tuple([ int( z / 2 ) for z in (crop, crop) ] ) 

            # crop and resize 
            dst = src.crop( ( centre[0]-crop2[0], centre[1]-crop2[1], centre[0]+crop2[0], centre[1]+crop2[1] ) )
            dst = dst.resize( (resize, resize), Image.ANTIALIAS )

            # save image if sufficient information content - based on mode count
            res = stats.mode( np.array( dst )[:,:,0].flatten() )
            if ( ( res.count[ 0 ] / resize **2 ) < 0.75 ):
                dst.save( out_pathname, 'JPEG', quality=95 )

    return 


def parseArguments(args=None):

    """
    parse command line argument
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description='data prep')
    parser.add_argument('image_path', action="store")
    parser.add_argument('out_path', action="store")

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()    

    # get and iterate images 
    path = os.path.join( args.image_path, '**' )
    images = glob.glob( os.path.join( path, '*_footprint.tif' ) )

    for image in images:

        # extract uid from image pathname
        uid = getUniqueId( image )
        if uid is not None:

            # create image chips
            getImageChips( image, os.path.join( args.out_path, uid ), crops=[1536,2048,3072,4096], resize=512 )

    return

# execute main
if __name__ == '__main__':
    main()
