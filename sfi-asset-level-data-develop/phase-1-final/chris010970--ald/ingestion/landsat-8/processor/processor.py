import os
import re
import sys
import osr
import numpy as np

import osr
from osgeo import gdal
from mask import Mask

# add utility functions
sys.path.append( os.path.join( os.path.dirname( sys.path[0] ), '../utility/' ) )
from ps import execute

class Processor:


    def __init__( self ):

        """
        Placeholder
        """

        # increase system memory usage
        os.environ['GDAL_CACHEMAX'] = '2048'
        gdal.UseExceptions()

        self._mask = Mask()
        return


    def getPansharpenImage( self, pathname ):

        """
        Placeholder
        """

        # get masked datasets
        code = None

        datasets = self.getMaskedDatasets ( os.path.dirname( pathname ) )
        if len( datasets ) == 3:

            # define pansharpen options
            args = [ os.path.join( os.path.dirname( pathname ), 'B8_merge.tif' ) ] + datasets + [ pathname ]
            options = [ '-co',  'PHOTOMETRIC=RGB', '-co', 'COMPRESS=DEFLATE', '-nodata', '0' ]

            # manage execution of gdal pansharpen
            out, err, code = execute( 'gdal_pansharpen.py', args + options )

        return code


    def getMaskedDatasets( self, path ):

        """
        Placeholder
        """

        # check qa image exists
        datasets = []

        qa_pathname = os.path.join( path, 'BQA_merge.tif' )
        if os.path.exists( qa_pathname ):

            arr = self._mask.generate( qa_pathname, out_pathname=os.path.join( path, 'mask.tif') )
            if arr is not None:

                # mask channel images
                channels = [ 'B4', 'B3' , 'B2' ]
                for channel in channels:

                    pathname = os.path.join( path, '{}_merge.tif'.format( channel ) )
                    out_pathname = pathname.replace( '_merge.tif', '_merge_mask.tif' ) 

                    # apply masking
                    arr = self._mask.apply( pathname, out_pathname=out_pathname )
                    if arr is not None:
                        datasets.append( out_pathname )

            else:

                # masking error
                print ( 'Masking error - skipping: {}'.format ( qa_pathname ) )

        else:

            # no qa file
            print ( 'File does not exist - skipping: {}'.format ( qa_pathname ) )

        return datasets

    
    def getImageChip( self, pathname, centroid, out_pathname, size=512, scale_min=2, scale_max=98 ):

        """
        Placeholder
        """

        # open pansharpen image
        ds = gdal.Open( pathname )
        if ds is not None:

            # transform latlon centroid coordinates to image srs
            coord_tx = self.getCoordinateTransform( ds )
            x, y, z = coord_tx.TransformPoint( centroid[ 0 ], centroid[ 1 ] )

            # convert image srs coordinates into pixel coordinates
            geo = ds.GetGeoTransform()
            pixel = ( round ( ( x - geo[ 0 ] ) / geo[ 1 ] ), round ( ( geo[ 3 ] - y ) / -geo[ 5 ] ) )

            x_offset = ( pixel[ 0 ] - ( size / 2 ) )
            y_offset = ( pixel[ 1 ] - ( size / 2 ) )

            # check chip window is valid
            if x_offset >= 0 and y_offset >= 0 and ( x_offset + size ) < ds.RasterXSize and ( y_offset + size ) < ds.RasterYSize: 

                # read in window
                r = ds.GetRasterBand(1).ReadAsArray( xoff=x_offset, yoff=y_offset, win_xsize=size, win_ysize=size )
                g = ds.GetRasterBand(2).ReadAsArray( xoff=x_offset, yoff=y_offset, win_xsize=size, win_ysize=size )
                b = ds.GetRasterBand(3).ReadAsArray( xoff=x_offset, yoff=y_offset, win_xsize=size, win_ysize=size )

                # validate goodness - number of black (masked) pixels 
                goodness = ( np.count_nonzero( r ) / ( size * size ) ) * 100.0
                if goodness > 95.0:

                    # get scale factors to convert from 16bit to 8bit 
                    scale = {   'r' : self.getScaleMinMax( r, ( scale_min, scale_max ) ),
                                'g' : self.getScaleMinMax( g, ( scale_min, scale_max ) ),
                                'b' : self.getScaleMinMax( b, ( scale_min, scale_max ) ) }

                    # compile translation options
                    options = [ '-of JPEG',
                                '-ot Byte',
                                '-srcwin {} {} {} {}'.format( x_offset, y_offset, size, size ),
                                '-scale_1 {} {}'.format ( scale[ 'r' ][ 0 ], scale[ 'r' ][ 1 ] ),
                                '-scale_2 {} {}'.format ( scale[ 'g' ][ 0 ], scale[ 'g' ][ 1 ] ),
                                '-scale_3 {} {}'.format ( scale[ 'b' ][ 0 ], scale[ 'b' ][ 1 ] ) ]

                    # create output directory if not exists
                    if not os.path.exists( os.path.dirname( out_pathname ) ):
                        os.makedirs( os.path.dirname( out_pathname ) )

                    # execute translation
                    options_str = ' '.join( options )
                    out = gdal.Translate( out_pathname, ds, options=options_str )

                    print ( 'Generated chip: {} {}'.format ( out_pathname, goodness ) )

                else:

                    # failed qc check
                    print ( 'Chip failed QC: {} {}'.format( out_pathname, goodness ) )        

            else:

                # report error
                print ( 'Invalid chip window {} {} {} {} ({} {}): {}'.format( x_offset, y_offset, x_offset + size, y_offset + size, 
                                                                                ds.RasterXSize, ds.RasterYSize, pathname ) )        
        else:

            # report error
            print ( ' Unable to open dataset: {}'.format( pathname ) )
    
        return 


    def getCoordinateTransform( self, ds ):

        """
        Placeholder
        """

        # get transformation from latlon to image srs
        image = osr.SpatialReference( wkt=ds.GetProjection() )

        aoi = osr.SpatialReference()
        aoi.ImportFromEPSG( 4326 )

        return osr.CoordinateTransformation( aoi, image )


    # compute percentiles from cdf
    def getScaleMinMax( self, arr, pcs ):

        """
        Placeholder
        """

        return np.percentile( arr[ arr != 0 ], pcs )
