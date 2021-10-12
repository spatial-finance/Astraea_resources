import os
import sys
import time
import random
import argparse
import pycountry
import pandas as pd
import numpy as np

from downloader import Downloader

sys.path.append( os.path.join( os.path.dirname( sys.path[0] ), '../utility/' ) )
from aoi import AoI


def parseArguments(args=None):

    """
    Placeholder
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description='sfi data prep')
    parser.add_argument('inventory_file', action="store")
    parser.add_argument('out_path', action="store")

    # start and end record indexes
    parser.add_argument('-sr', '--start_record',
                    type=int,
                    help='start record',
                    default='0')

    parser.add_argument('-er', '--end_record',
                    type=int,
                    help='end record',
                    default='0')

    # start and end dates
    parser.add_argument('-sd', '--start_date',
                    help='start date',
                    default='2017-01-01')

    parser.add_argument('-ed', '--end_date',
                    help='end date',
                    default='2017-03-01')

    return parser.parse_args(args)


def main():

    """
    Placeholder
    """

    # parse arguments
    args = parseArguments()

    # load spreadsheet
    df = pd.read_excel( args.inventory_file, 
        sheet_name=os.path.splitext( os.path.basename( args.inventory_file ) )[ 0 ] )

    # define start and end records
    start_record = 0; end_record = len( df )
    if args.end_record > args.start_record:
        start_record = args.start_record; end_record = args.end_record

    print ( 'processing records {} -> {}'.format( start_record, end_record ) )
    print ( 'date range {} -> {}'.format( args.start_date, args.end_date ) )

    # create downloader object
    key_file = '/home/sac/Downloads/arkham-255409-c59a52d8653f.json'
    project_id = 'arkham-255409'

    prefixes = [ 'manifest.safe', '_TCI.jp2' ]
    obj = Downloader( key_file, project_id, prefixes ) 
    scenes = obj.queryBucket( start_date=args.start_date, end_date=args.end_date, max_cloudcover=5 )

    print ( 'found {} candidate scenes'.format( len( df ) ) )

    # loop through asset records
    aoi = AoI(); max_tries = 3
    for idx, row in df.iterrows():

        # if record within index constraints
        if idx >= start_record and idx <= end_record:

            if isinstance( row.accuracy, str ) and row.accuracy.lower() == 'exact':

                # get bounding box centred on asset location
                bbox = aoi.getBoundingBox( ( row.longitude, row.latitude ), distance=3000 )
                print ( 'processing record {}: {} {}'.format( idx, row.uid, bbox.bounds ) )

                # google doesn't support automated requests (!)
                tries = 1
                while tries <= max_tries:

                    try:
                        # retrieve images aligned with constraints            
                        out_path = os.path.join( args.out_path, row.uid )
                        obj.process( scenes, bbox.bounds, out_path )
                        break
                    
                    except:
                        # retry after random wait
                        print ( 'retry after exception {}: {}'.format( tries, max_tries ) )
                        tries += 1
                        time.sleep ( random.randrange( 10 ) )

            else:

                # ignore assets with imprecise location
                print ( 'skipping record (non-exact location) {}'.format( idx ) )

    return

# execute main
if __name__ == '__main__':
    main()

