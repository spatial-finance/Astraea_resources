import os
import sys
import xmltodict
import numpy as np
import pandas as pd
import geopandas as gpd

import pyrasterframes
import pyrasterframes.rf_ipython

from pyspark import SparkContext
from pyrasterframes.rasterfunctions import *
from pyrasterframes.utils import create_rf_spark_session

from shapely import wkt

import osr
from datetime import datetime
from osgeo import gdal, gdal_array

# add utility functions
sys.path.append( os.path.join( os.path.dirname( sys.path[0] ), '../utility/' ) )
from fs import getFileList


class Downloader:

    def __init__( self, uri, band_sets, max_chips=10, tile_dimensions=(512,512), memory='5g', min_valid_pct=85 ):

        """
        placeholder
        """

        # start session and set configuration
        self._spark = create_rf_spark_session()

        self._spark.conf.set( 'spark.executor.memory', memory )
        self._spark.conf.set( 'spark.driver.memory', memory )

        # load catalog
        self._cat = self._spark.read.format( uri ).load().repartition(20)
        self._tile_dimensions = tile_dimensions

        self._band_sets = band_sets
        self._min_valid_pct = min_valid_pct
        self._max_tries = 3
        self._max_chips = 4

        return


    def process( self, aoi, out_path, **kwargs ):

        """
        placeholder
        """

        # retrieve subset of scenes from catalog
        scenes = self.filterCatalog( **kwargs )
        scenes = scenes.crossJoin( self.getAoi( aoi ) )

        # filter scenes dependent on aoi intersection
        scenes = scenes.withColumn('geom', st_geometry( scenes.bounds_wgs84 ) )
        scenes = scenes.filter(st_intersects('geom', 'aoi')) 

        # get scenes satisfying constraints
        df = scenes.toPandas()
        for idx, row in df.iterrows():

            self.getScene( row, out_path )

            # evaluate number of successfully downloaded scenes
            files = getFileList( out_path, 'B8_merge.tif' )
            if len( files ) > self._max_chips:

                # maximum chips found
                print ( 'maximum scenes {} downloaded: {}'. format ( self._max_chips, out_path ) )
                break

        return


    def filterCatalog( self, **kwargs ):

        """
        placeholder
        """

        scenes = self._cat        

        # start date
        start_date = kwargs.pop( 'start_date', None )
        if start_date is not None:
            scenes = scenes.filter( scenes.acquisition_date >= start_date )

        # end date
        end_date = kwargs.pop( 'end_date', None )
        if end_date is not None:
            scenes = scenes.filter( scenes.acquisition_date <= end_date )

        # cloud cover
        max_cloud_cover = kwargs.pop( 'max_cloudcover', None )
        if max_cloud_cover is not None and 'cloud_cover_pct' in scenes.columns:
            scenes = scenes.filter( max_cloud_cover >= scenes.cloud_cover_pct )

        # processing level
        processing_level = kwargs.pop( 'processing_level', None )
        if processing_level is not None and 'processing_level' in scenes.columns:
            scenes = scenes.filter( processing_level == scenes.processing_level )

        return scenes


    def getAoi( self, aoi ):

        """
        placeholder
        """

        # create geo dataframe
        df = pd.DataFrame( { 'aoi' : [ aoi ] }  )
        df[ 'aoi' ] = df[ 'aoi' ].apply(wkt.loads)

        # convert to spark dataframe
        gdf = gpd.GeoDataFrame( df, geometry='aoi')
        
        return self._spark.createDataFrame(gdf)


    def getScene( self, row, out_path ):

        """
        placeholder
        """

        # construct unique output path                
        dt = datetime.strftime ( row[ 'acquisition_date' ], "%Y%m%d_%H%M%S" )
        scene_path = os.path.join( out_path, dt )

        # check path already exists
        if not os.path.exists( scene_path ):

            # write dataframe row to file
            self.writeSceneInfo( row, scene_path  )

            for band_set in self._band_sets:

                # retrieve raster band images
                files = self.getRasterBands( row, scene_path, band_set )
                if len( files ) != len( band_set ):
                    break
        else:

            # path exists - skipping scene
            print ( 'skipping scene: {}'. format ( scene_path ) )

        return


    def getRasterBands( self, row, scene_path, band_set ):

        """
        placeholder
        """

        files = []

        # reformat single row dataframe
        df = row.to_frame().T
        df = df[ [ 'acquisition_date', 'aoi' ] + band_set ]

        raster_df = self._spark.read.raster( self._spark.createDataFrame( df ), 
                            catalog_col_names=band_set,
                            tile_dimensions=self._tile_dimensions )

        # compile additional columns
        tiles_df = raster_df.withColumn( 'extent', rf_extent( band_set[ 0 ] ) )
        tiles_df = tiles_df.withColumn( 'crs', rf_crs( band_set[ 0 ] ) )
        tiles_df = tiles_df.withColumn( 'geom', st_reproject( rf_geometry( band_set[ 0 ] ), 
                                        rf_crs( band_set[ 0 ] ), lit('EPSG:4326')) )
                    
        # get tiles coincident with aoi
        tiles_df = tiles_df.filter(st_intersects('geom', 'aoi'))
        if tiles_df.count() > 0:
        
            # save tiles to file
            tiles = tiles_df.rdd.collect()
            for idx, band in enumerate( band_set ):

                # write tiles to file
                images, valid_pct = self.writeTiles( tiles, scene_path, band )

                if idx == 0 and valid_pct < self._min_valid_pct:
                    print ( 'ignored scene: {} ({})'.format( scene_path, valid_pct ) )
                    break

                # merge into single image
                pathname = os.path.join( scene_path, '{}_merge.tif'.format( band ) )
                gdal.Warp( pathname, images )

                # record pathname into list
                print ( 'saved image: {}'. format ( pathname ) )
                files.append( pathname )

        else:

            # path exists - skipping scene
            print ( 'no scenes found: {}'. format ( scene_path ) )

        return files


    def writeSceneInfo( self, row, out_path ):

        """
        placeholder
        """

        # create directory structure if required
        if not os.path.exists( out_path ):
            os.makedirs( out_path )

        # transform dict to xml schema
        schema = { 'root' : dict( row ) }
        out = xmltodict.unparse( schema, pretty=True )

        # write serialized xml schema to file
        with open( os.path.join( out_path, 'info.xml' ), 'w+') as file:
            file.write(out)

        return


    def writeTiles( self, records, out_path, band ):

        """
        placeholder
        """

        # create directory structure if required
        if not os.path.exists( out_path ):
            os.makedirs( out_path )

        # for each tile record
        images = []; non_zero = count = 0
        for idx, record in enumerate( records ):

            # create geotiff 
            driver = gdal.GetDriverByName( 'GTiff' )
            pathname = os.path.join( out_path, '{}_{}.tif'.format( band, idx ) )

            # get dimensions and data type
            rows, cols = record[ band ].tile.cells.shape
            dtype = gdal_array.NumericTypeCodeToGDALTypeCode( np.dtype(record[ band ].tile.cells.dtype.name ) )

            # get resolution
            xres = ( record.extent[ 'xmax' ] - record.extent[ 'xmin' ] ) / cols
            yres = ( record.extent[ 'ymax' ] - record.extent[ 'ymin' ] ) / rows
            
            # get srs object
            out_ds = driver.Create( pathname, xsize=cols, ysize=rows, bands=1, eType=dtype, options=[ 'TILED=YES', 'COMPRESS=DEFLATE'] )
            if out_ds is not None:

                # get srs object
                srs = osr.SpatialReference()
                srs.ImportFromProj4( record.crs[ 'crsProj4' ] )

                # copy over projection and geo transform
                out_ds.SetProjection( srs.ExportToWkt() )
                out_ds.SetGeoTransform( 
                            [ record.extent[ 'xmin'], xres, 0, record.extent[ 'ymax' ], 0, -yres ] )

                # copy pixel data and save file
                out_ds.GetRasterBand(1).WriteArray( np.uint16( record[ band ].tile.cells ) )
                out_ds = None

                # update non zero stats 
                non_zero += np.count_nonzero( record[ band ].tile.cells )
                count += ( rows * cols )

            # add file if exists
            if os.path.exists( pathname ):
                images.append( pathname )

        return images, ( non_zero / count ) * 100
