import os
import sys
import argparse
import pycountry
import pandas as pd
import numpy as np

from downloader import Downloader

# add utility functions
sys.path.append( os.path.join( os.path.dirname( sys.path[0] ), '../utility/' ) )
from aoi import AoI




def getCatalogCheck( row, year )

    """
    optionally check if sufficient chips already available for a given year
    """

    condition = False

    # quick catalog check
    chip_count = row [ 'l8_{}'.format( year ) ]
    if chip_count < min_chips:
        condition = True

    return condition


def parseArguments(args=None):

    """
    parse arguments from command line
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

    parser.add_argument('--catalog_check', 
                        default=False, 
                        action='store_true', 
                        help='catalog check' )

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()

    # load spreadsheet
    df = pd.read_csv( args.inventory_file )

    # define start and end records
    start_record = 0; end_record = len( df )
    if args.end_record > args.start_record:
        start_record = args.start_record; end_record = args.end_record

    print ( 'processing records {} -> {}'.format( start_record, end_record ) )
    print ( 'date range {} -> {}'.format( args.start_date, args.end_date ) )

    # create downloader object
    # band_sets = [   [ 'B2' ],
    #                [ 'B3', 'B4', 'BQA' ],
    #                [ 'B8' ]
    # ]
    band_sets = [ [ 'B5', 'B6', 'B7', 'B10', 'B12' ] ]

    obj = Downloader( 'aws-pds-l8-catalog', band_sets, tile_dimensions=(512,512) )

    # loop through asset records
    aoi = AoI()
    for idx, row in df.iterrows():

        # if record within index constraints
        if idx >= start_record and idx <= end_record:

            if isinstance( row.accuracy, str ) and row.accuracy.lower() == 'exact':

                # optional catalog check
                process = True
                if args.catalogCheck is True:
                    process = getCatalogCheck( row, args.start_date[ 0:4 ] )

                if process is True:

                    # get bounding box centred on asset location
                    bbox = aoi.getBoundingBox( ( row.longitude, row.latitude ), distance=10000 )
                    try:

                        if scenes is True:
                            scenes = glob.glob( os.path.join( args.out_path, row.uid ), 'B4_merge_mask.tif'

                        # retrieve images aligned with constraints
                        print ( 'processing record {}: {} {}'.format( idx, row.uid, bbox.bounds ) )
                        obj.process( bbox.wkt, os.path.join( args.out_path, row.uid ), 
                                    start_date=args.start_date, end_date=args.end_date, max_cloudcover=5, scenes=scenes )
                    except:
                        pass

                else:

                    # failed conditional checks
                    print ( 'skipping record (failed conditions) {} {}'.format( row.uid, idx ) )

            else:

                # ignore assets with imprecise location
                print ( 'skipping record (non-exact location) {} {}'.format( row.uid, idx ) )

    return

# execute main
if __name__ == '__main__':
    main()

