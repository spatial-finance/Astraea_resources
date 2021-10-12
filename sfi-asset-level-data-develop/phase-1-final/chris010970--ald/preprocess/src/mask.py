import os
import glob
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from osgeo import gdal, ogr
from utility.aoi import AoI
from utility.dp import getUniqueId


def applyMaskToImage( src_ds, dst_ds, mask ):

    """
    set values in image to zero
    """

    # apply mask
    mask = mask == 255
    mask = ~mask

    for idx in range ( src_ds.RasterCount ):

        data = src_ds.GetRasterBand( idx + 1 ).ReadAsArray()
        data[ mask ] = round( np.mean(data) ).astype( np.uint8 )

        dst_ds.GetRasterBand( idx + 1 ).WriteArray( data )

    return


def getFootprintImage( obj, out_pathname ):

    """
    set values in image to zero
    """

    # read mask into numpy array
    ds = gdal.Open( obj[ 'mask' ] )
    mask = ds.ReadAsArray()
    ds = None

    # read source
    src_ds = gdal.Open( obj[ 'image' ] )
    driver = gdal.GetDriverByName('GTiff')

    # create destination
    dst_ds = driver.Create( out_pathname, 
                            src_ds.RasterXSize, 
                            src_ds.RasterYSize, 
                            src_ds.RasterCount, 
                            gdal.GDT_Byte, 
                            options=["TILED=YES", "COMPRESS=DEFLATE"] )

    dst_ds.SetProjection( src_ds.GetProjection() )
    dst_ds.SetGeoTransform( src_ds.GetGeoTransform() )

    # apply mask
    applyMaskToImage( src_ds, dst_ds, mask )
    src_ds = None; dst_ds = None

    return


def addPolygonToMask( ds, pathname ):

    """
    burn shapefile into image buffer
    """

    # load land polygons shapefile
    fid = ogr.Open( pathname )
    layer = fid.GetLayer()

    # rasterize vector sub-region
    error = gdal.RasterizeLayer(    ds,            # output to our new dataset
                                    [1],           # output to our new dataset's first band
                                    layer,         # rasterize this layer
                                    None, None,    # don't worry about transformations since we're in same projection
                                    [255],           # burn value 0
                                    ['ALL_TOUCHED=TRUE' ]   # rasterize all pixels touched by polygons
                                    )
    ds.FlushCache()

    return


def getGeocodedMask( image_pathname, footprints, options, out_path ):

    """
    get footprint mask
    """

    out = None

    # open image file
    in_ds = gdal.Open( image_pathname )
    if in_ds is not None:

        # reproject to epsg
        out_pathname = os.path.join( out_path, os.path.basename ( image_pathname ).replace( '.tif', '-utm.tif' ) )
        out_ds = gdal.Warp( out_pathname, in_ds, options=options )

        # create mask with lossless compression
        mask_pathname = out_pathname.replace( '.tif', '-mask.tif' )
        driver = gdal.GetDriverByName('GTiff')

        mask_ds = driver.Create(    mask_pathname, 
                                    out_ds.RasterXSize, 
                                    out_ds.RasterYSize, 
                                    1, 
                                    gdal.GDT_Byte, 
                                    options=[ 'TILED=YES', 'COMPRESS=DEFLATE' ]  )

        if mask_ds is not None:

            # copy projection
            mask_ds.SetProjection( out_ds.GetProjection() )
            mask_ds.SetGeoTransform( out_ds.GetGeoTransform() )  
            mask_ds.GetRasterBand(1).Fill( 0 )

            # create footprint mask aligned with image
            for footprint in footprints:
                addPolygonToMask( mask_ds, footprint )

            out = { 'image' : out_pathname, 'mask' :mask_pathname }
            mask_ds = None

        # flush buffers
        out_ds = None

    return out


def parseArguments(args=None):

    """
    parse command line argument
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description='data prep')
    parser.add_argument('inventory_file', action="store")
    parser.add_argument('image_path', action="store")
    parser.add_argument('footprint_path', action="store")
    parser.add_argument('out_path', action="store")

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()    
    df = pd.read_csv( args.inventory_file )
    aoi = AoI()

    # iterate across images
    images = glob.glob( os.path.join( os.path.join( args.image_path, '**' ), '*.tif' ) )
    for image in images:

        # get unique id from pathname
        uid = getUniqueId( image )
        if uid is not None:

            # get corresponding record in data table
            record = df[ df[ 'uid' ] == uid ]
            if len( record ) == 1:

                # get geojson files for uid
                path = os.path.join( args.footprint_path, '{}'.format( uid ) )
                footprints = glob.glob( os.path.join( path, '{}*.geojson' ).format( uid ) )
                if len( footprints ) > 0:

                    # create output path
                    out_path = os.path.join( args.out_path, uid )
                    if not os.path.exists( out_path ):
                        os.makedirs( out_path )

                    # get output pathname
                    out_pathname = os.path.join( out_path, '{uid}_footprint.tif'.format ( uid=uid ) )
                    if not os.path.exists( out_pathname ):

                        print ( 'processing: {}'.format( out_pathname ) )

                        # get footprint binary mask - if sufficient data available
                        epsg = aoi.getEpsg( ( record[ 'longitude' ].iloc[0], record[ 'latitude'].iloc[0] ) )
                        out = getGeocodedMask(  image, 
                                                footprints, 
                                                '-of GTiff -t_srs epsg:{epsg} -co TILED=YES -co COMPRESS=DEFLATE'.format( epsg=epsg ),
                                                out_path )
                                                
                        # created utm masked image
                        if out is not None:                    
                            getFootprintImage( out, out_pathname )

    return

# execute main
if __name__ == '__main__':
    main()
