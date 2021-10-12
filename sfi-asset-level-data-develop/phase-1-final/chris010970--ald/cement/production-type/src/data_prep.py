import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd

from utility.dp import getUniqueId
from sklearn.model_selection import train_test_split


def createDataset( df, out_path ):

    """
    crop image at various scales and resize to common shape
    """

    # create out path
    if not os.path.exists( out_path ):
        os.makedirs( out_path )

    # iterate through rows
    for idx, row in df.iterrows():

        # copy image into subset sub-directory
        filename = os.path.basename ( row[ 'image' ] )
        shutil.copy( row[ 'image' ], os.path.join( out_path, filename ) )

    # remove path from image column
    df2 = df.copy()
    df2[ 'image' ] = df2[ 'image' ].apply( lambda x : os.path.basename( x ) )

    # return filtered column list
    return df2[ [ 'image', 'uid', 'status', 'latitude', 'longitude', 'plant_type', 'kiln_type_1', 'capacity' ] ]


def getImageDataFrame( image_path, crops=[2048] ):

    """
    placeholder
    """

    # get image chips selected by crop size
    sub_dirs = glob.glob( os.path.join( image_path, '*' ) )
    data = { 'image' : [], 'uid' : [] }

    for sub_dir in sub_dirs:

        images = glob.glob( os.path.join( sub_dir, '*.jpg' ) )
        images.sort()

        data[ 'image' ].extend ( images[ -2: ] )

    # get uids and convert dict to dataframe
    data[ 'uid' ] = [ getUniqueId( image ) for image in data[ 'image' ] ]
    return pd.DataFrame.from_dict( data )


def parseArguments(args=None): 

    """
    parse arguments from command line
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description='data prep')
    parser.add_argument('inventory_file', action="store")
    parser.add_argument('image_path', action="store")
    parser.add_argument('out_path', action="store")

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()    
    df = pd.read_csv( args.inventory_file )

    # locate records with defined plant type and exact location
    df = df[ ( df[ 'kiln_type_1' ].notna() ) & ( df[ 'accuracy' ] == 'Exact' ) ]
    df[ 'kiln_type_1' ] = df[ 'kiln_type_1' ].str.lower()

    df_image = getImageDataFrame( args.image_path, crops=[ 1536, 2048, 3072, 4096 ] )

    # get category subset 
    for label in [ 'wet', 'dry' ]:

        # randomly split subset into train and test
        df_subset = df[ df[ 'kiln_type_1' ] == label ]
        df_train, df_test = train_test_split( df_subset, test_size=0.2 )

        # merge data frames
        df_train = pd.merge( df_train, df_image, on='uid', how='inner' )
        df_test = pd.merge( df_test, df_image, on='uid', how='inner' )

        print ( '{}: Training size {} - Test size {}'.format ( label, len( df_train ), len( df_test ) ) )

        # generate train subset
        out_path = os.path.join( os.path.join( args.out_path, 'train' ), label )
        df_train = createDataset( df_train, out_path )
        df_train.to_csv( os.path.join( args.out_path, 'train.csv' ), index=False )

        # generate test subset
        out_path = os.path.join( os.path.join( args.out_path, 'test' ), label )
        df_test = createDataset( df_test, out_path )
        df_test.to_csv( os.path.join( args.out_path, 'test.csv' ), index=False )

    return


# execute main
if __name__ == '__main__':
    main()
