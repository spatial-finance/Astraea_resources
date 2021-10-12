import os
import re
import sys
import osr
import shutil
import numpy as np

import osr
from osgeo import gdal

# add utility functions
sys.path.append( os.path.join( os.path.dirname( sys.path[0] ), '../utility/' ) )
from fs import getFileList

class Processor:


    def __init__( self ):

        """
        Placeholder
        """

        # increase system memory usage
        os.environ['GDAL_CACHEMAX'] = '2048'
        gdal.UseExceptions()

        self._resolution = 10.0
        return


    def getImageChip( self, path, centroid, out_pathname, size=512, overwrite=False ):

        """
        Placeholder
        """

        # file not created or overwrite
        if not os.path.exists( out_pathname ):

            # get scenes downloaded for datetime
            images = self.clipScenes( getFileList( path, '_TCI.jp2' ), centroid, size )             
            # images = [ '/data/test/ALB0003/20170820_093607/34TDL/T34TDL_20170820T093041_TCI-chip.tif' ]

            if len ( images ) > 0:

                # get scene srs
                ds = gdal.Open( images [ 0 ] )
                coord_tx = self.getCoordinateTransform( ds )

                # transform latlon centroid coordinates to image srs
                x, y, z = coord_tx.TransformPoint( centroid[ 0 ], centroid[ 1 ] )
                x = round( x ); y = round ( y )

                distance = ( size / 2 ) * self._resolution

                x0 = x - distance; y0 = y - distance
                x1 = x + distance; y1 = y + distance

                print ( x0, y0, x1, y1 )

                # compile translation options
                options = [ '-of MEM',
                            '-t_srs {}'.format( ds.GetProjection() ),
                            '-te {} {} {} {}'.format( x0, y0, x1, y1 ),
                            '-tr {} {}'.format ( self._resolution, self._resolution ),
                            '-srcnodata 0'
                            ]

                # combine images into final output
                options_str = ' '.join( options )
                warp_ds = gdal.Warp( '', images, options=options_str )

                # validate goodness - number of black (masked) pixels 
                goodness = ( np.count_nonzero( warp_ds.GetRasterBand(1).ReadAsArray() ) / ( size * size ) ) * 100.0
                if goodness > 95.0:

                    # spit in-memory image to jpg file
                    if not os.path.exists ( os.path.dirname( out_pathname ) ):
                        os.makedirs( os.path.dirname( out_pathname ) )

                    gdal.Translate( out_pathname, warp_ds, options='-of JPEG' )
                    print ( 'Generated chip: {} {}'.format ( out_pathname, goodness ) )

                else:

                    # failed qc check
                    print ( 'Chip failed QC: {} {}'.format( out_pathname, goodness ) )        

                # housekeeping
                warp_ds = None
                ds = None

        return 


    def clipScenes( self, scenes, centroid, size ):

        """
        Placeholder
        """

        # for each scene
        images = []
        for scene in scenes:

            # get scene srs
            ds = gdal.Open( scene )
            coord_tx = self.getCoordinateTransform( ds )

            # transform latlon centroid coordinates to image srs
            x, y, z = coord_tx.TransformPoint( centroid[ 0 ], centroid[ 1 ] )
            x = round( x ); y = round ( y )

            # convert image srs coordinates into pixel coordinates
            geo = ds.GetGeoTransform()
            pixel = ( round ( ( x - geo[ 0 ] ) / geo[ 1 ] ), round ( ( geo[ 3 ] - y ) / -geo[ 5 ] ) )

            pad = 10; pad2 = pad * 2

            x_offset = pixel[ 0 ] - ( ( size / 2 ) + pad )
            y_offset = pixel[ 1 ] - ( ( size / 2 ) + pad )

            # next interation if no intersection
            if ( x_offset + ( size + pad2 ) ) < 0 or x_offset > ds.RasterXSize or ( y_offset + ( size + pad2 ) ) < 0 or y_offset > ds.RasterYSize:
                continue

            # compile translation options
            options = [ '-of GTiff',
                        '-srcwin {} {} {} {}'.format( x_offset, y_offset, size + pad2, size + pad2 ),
                        ]

            # execute translation
            options_str = ' '.join( options )
            pathname = scene.replace( '.jp2', '-chip.tif')

            print ( options_str )
            out_ds = gdal.Translate( pathname, ds, options=options_str )
            out_ds = None

            images.append( pathname )

        return images


    def getCoordinateTransform( self, ds ):

        """
        Placeholder
        """

        # get transformation from latlon to image srs
        image = osr.SpatialReference( wkt=ds.GetProjection() )

        aoi = osr.SpatialReference()
        aoi.ImportFromEPSG( 4326 )

        return osr.CoordinateTransformation( aoi, image )

