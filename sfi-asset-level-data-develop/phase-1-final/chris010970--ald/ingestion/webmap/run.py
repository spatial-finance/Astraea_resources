import os
import sys
import random
import argparse
import pycountry
import numpy as np
import pandas as pd

from utility.aoi import AoI
from downloader import Downloader
from global_land_mask import globe
from tiler import MercatorTiler, SlippyTiler


def getRandomLocations( longitude, latitude, samples ):

    """
    get randomised points around asset location
    """

    # iterate number of samples
    pts = []
    for idx in range( samples ):

        while True:
        
            # loop until land location found
            pt = { 'longitude': longitude + random.uniform( -1.0, 1.0 ), 'latitude': latitude + random.uniform( -1.0, 1.0 ) }
            if globe.is_land( pt[ 'latitude' ], pt[ 'longitude' ] ):
                break

        # append to list
        pts.append ( pt )

    return pts


def parseArguments(args=None):

    """
    parse arguments
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
    #                default='891')

    parser.add_argument('-er', '--end_record',
                    type=int,
                    help='end record',
                    default='4000')

    parser.add_argument('-z', '--zoom',
                    type=int,
                    help='zoom',
                    default='17')

    parser.add_argument('-b', '--background',
                    type=int,
                    help='background',
                    default='0')

    return parser.parse_args(args)


def main():

    """
    Placeholder
    """

    # parse arguments
    args = parseArguments()
    args.distance = 1000

    # load spreadsheet
    # df = pd.read_excel( args.inventory_file, 
    #    sheet_name=os.path.splitext( os.path.basename( args.inventory_file ) )[ 0 ] )
    df = pd.read_csv( args.inventory_file )

    # define start and end records
    start_record = 0; end_record = len( df )
    if args.end_record > args.start_record:
        start_record = args.start_record; end_record = args.end_record

    print ( 'processing records {} -> {}'.format( start_record, end_record ) )

    #obj = Downloader( 'https://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png256?access_token=pk.eyJ1Ijoib3BlbnN0cmVldG1hcCIsImEiOiJjaml5MjVyb3MwMWV0M3hxYmUzdGdwbzE4In0.q548FjhsSJzvXsGlPsFxAQ',
    #                    options='-expand rgb') 

    obj = Downloader( 'https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', img_format='jpg' )

    # loop through asset records
    aoi = AoI()
    for idx, row in df.iterrows():

        # if record within index constraints
        if idx >= start_record and idx <= end_record:

            if isinstance( row.accuracy, str ) and row.accuracy.lower() == 'exact':

                # get random locations
                pts = [ { 'longitude' : row.longitude, 'latitude' : row.latitude } ] 
                if args.background > 0:
                    pts = getRandomLocations( row.longitude, row.latitude, args.background )

                # iterate points
                for idx_pt, pt in enumerate( pts ):
                    
                    # get bounding box centred on point location
                    bbox = aoi.getBoundingBox( ( pt[ 'longitude' ], pt[ 'latitude' ] ), distance=args.distance )                    
                    try:

                        # get filename - adjusted for background option
                        filename = '{}_{}_{}.tif'.format ( row.uid, args.zoom, args.distance )
                        if args.background:
                            filename = filename.replace( '.tif', '_{}_bg.tif'.format( idx_pt ) )

                        out_pathname = os.path.join( os.path.join( args.out_path, row.uid ), filename )
                        if not os.path.exists( out_pathname ):

                            # retrieve images aligned with constraints            
                            print ( 'processing record {}: {} {}'.format( idx, row.uid, bbox.bounds ) )
                            obj.process( bbox.bounds, args.zoom, out_pathname )

                    except Exception as e:
                        print ( 'Exception: {}'.format( str( e ) ) )

            else:

                # ignore assets with imprecise location
                print ( 'skipping record (non-exact location) {} {}'.format( row.uid, idx ) )

    return


# execute main
if __name__ == '__main__':
    main()
