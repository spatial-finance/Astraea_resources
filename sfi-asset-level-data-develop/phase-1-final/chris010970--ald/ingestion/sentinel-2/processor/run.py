import os
import re
import sys
import argparse
import pandas as pd

from processor import Processor

# add utility functions
sys.path.append( os.path.join( os.path.dirname( sys.path[0] ), '../utility/' ) )
from fs import getPathList
from dp import getDateTimeString, getUniqueId
from aoi import AoI


def parseArguments(args=None):

    """
    Placeholder
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description='sentinel-2 processor')
    parser.add_argument('inventory_file', action="store")
    parser.add_argument('root_path', action="store")
    parser.add_argument('out_path', action="store")

    # image chip size
    parser.add_argument('-c', '--chip_size',
                    type=int,
                    help='chip size',
                    default='512')

    # overwrite
    parser.set_defaults(overwrite=False)
    parser.add_argument('--overwrite',
                        help='overwrite',
                        dest='overwrite', action='store_true' )

    return parser.parse_args(args)


def main():

    """
    Placeholder
    """

    # parse command line arguments
    args = parseArguments()
    aoi = AoI(); processor = Processor()

    # load spreadsheet
    df = pd.read_excel( args.inventory_file, 
        sheet_name=os.path.splitext( os.path.basename( args.inventory_file ) )[ 0 ] )

    # extract data folders
    pathlist = getPathList( args.root_path, '[0-9]{8}_[0-9]{6}$' )
    for path in pathlist:

        # extract unique details from pathname
        uid = getUniqueId( path )
        dt = getDateTimeString( path )

        if uid is not None and dt is not None:

            # retrieve record from spreadsheet
            record = df.loc[ df[ 'uid' ] == uid ]
            if record is not None:

                try:
                    # merge TCI images into single image chip
                    out_pathname = os.path.join( os.path.join( args.out_path, uid ), '{}-{}.jpg'.format( uid, dt ) )
                    processor.getImageChip( path, ( float ( record[ 'longitude' ] ), float( record[ 'latitude' ] ) ), out_pathname, size=512 )
                    
                except:
                    pass

    return

# execute main
if __name__ == '__main__':
    main()
