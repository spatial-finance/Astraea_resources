import os
import sys
import xmltodict
import numpy as np
import pandas as pd

from google.cloud import bigquery, storage
from google.oauth2 import service_account

import osr
from datetime import datetime
from osgeo import gdal, gdal_array

# add utility functions
sys.path.append( os.path.join( os.path.dirname( sys.path[0] ), '../utility/' ) )
from fs import getFileList


class Downloader:

    def __init__( self, key_file, project_id, prefixes, max_scenes=10 ):

        """
        placeholder
        """

        # need service account to access big-query
        self._key_file=key_file
        self._project_id=project_id

        # setup configuration
        self._prefixes = prefixes
        self._max_scenes = max_scenes

        return


    def process( self, scenes, bounds, out_path ):

        """
        placeholder
        """

        # get client connection to s2 bucket
        client = storage.Client()
        bucket = client.get_bucket("gcp-public-data-sentinel-2")

        for idx, row in scenes.iterrows():

            # next interation if no intersection
            if row[ 'east_lon' ] < bounds[ 0 ] or row[ 'west_lon' ] >  bounds[ 2 ] or row[ 'south_lat' ] > bounds[ 1 ] or row[ 'north_lat' ] < bounds[ 3 ]:
                continue

            # evaluate number of successfully downloaded scenes
            files = getFileList( out_path, 'manifest.safe' )
            if len( files ) > self._max_scenes:

                # maximum chips found
                print ( 'maximum scenes {} downloaded: {}'. format ( self._max_scenes, out_path ) )
                break

            # construct scene path
            dt = datetime.strptime( row[ 'sensing_time' ], '%Y-%m-%dT%H:%M:%S.%fZ' )
            scene_path = os.path.join( os.path.join( out_path, dt.strftime( '%Y%m%d_%H%M%S' ) ), row[ 'mgrs_tile' ] )

            # get filelist in bucket prefix subfolder
            prefix = row[ 'base_url' ][ len( 'gs://gcp-public-data-sentinel-2/' ) : ]

            blobs = bucket.list_blobs(prefix=prefix, delimiter=None)
            for blob in blobs:

                # apply match to bucket files
                d = self.blob2dict( blob )
                if any( f in d[ 'name' ] for f in self._prefixes ):

                    if not os.path.exists( scene_path ):
                        os.makedirs( scene_path )

                    print ( 'downloading {} -> {}'.format( d[ 'name' ], scene_path ) )

                    # download bucket file to local directory
                    pathname = os.path.join( scene_path, os.path.basename( d[ 'name' ] ) )
                    if not os.path.exists ( pathname ):

                        with open( pathname, 'w+b' ) as z:
                            blob.download_to_file( z )
    
        return


    def queryBucket( self, **kwargs ):

        """
        placeholder
        """

        # get client connection
        credentials = service_account.Credentials.from_service_account_file( self._key_file )
        client = bigquery.Client(credentials=credentials, project=self._project_id )

        # pop out query conditions
        start_date = kwargs.pop( 'start_date', '2018-01-01' )
        end_date = kwargs.pop( 'end_date', '2018-12-31' )
        max_cloudcover = kwargs.pop( 'max_cloudcover', 5.0 )

        # construct and execute query
        # query = client.query("""
        #             SELECT * FROM `bigquery-public-data.cloud_storage_geo_index.sentinel_2_index`
        #                 WHERE 
        #                 CAST(SUBSTR(sensing_time, 1, 10) AS DATE) >= CAST('{}' AS DATE) AND 
        #                 CAST(SUBSTR(sensing_time, 1, 10) AS DATE) < CAST('{}' AS DATE) AND
        #                 NOT ( east_lon < {} OR west_lon > {} OR south_lat > {} OR north_lat < {} ) AND
        #                 CAST(cloud_cover AS FLOAT64) <= {} ORDER BY sensing_time
        #             """.format( start_date, end_date, bounds[ 0 ], bounds[ 2 ], bounds[ 1 ], bounds[ 3 ], max_cloudcover ))

        # construct and execute query
        query = client.query("""
                    SELECT * FROM `bigquery-public-data.cloud_storage_geo_index.sentinel_2_index`
                        WHERE 
                        CAST(SUBSTR(sensing_time, 1, 10) AS DATE) >= CAST('{}' AS DATE) AND 
                        CAST(SUBSTR(sensing_time, 1, 10) AS DATE) < CAST('{}' AS DATE) AND
                        CAST(cloud_cover AS FLOAT64) <= {} ORDER BY sensing_time
                    """.format( start_date, end_date, max_cloudcover ))

        results = query.result()

        return results.to_dataframe()


    def blob2dict( self, blob ):

        """
        Converts a google.cloud.storage.Blob (which represents a storage object) to context format (GCS.BucketObject).
        """

        # unpack blob into dictionary
        return {
            'name': blob.name,
            'bucket': blob.bucket.name,
            'contentType': blob.content_type,
            'timeCreated': blob.time_created,
            'timeUpdated': blob.updated,
            'timeDeleted': blob.time_deleted,
            'size': blob.size,
            'MD5': blob.md5_hash,
            'ownerID': '' if not blob.owner else blob.owner.get('entityId', ''),
            'CRC32c': blob.crc32c,
            'encryptionAlgorithm': blob._properties.get('customerEncryption', {}).get('encryptionAlgorithm', ''),
            'encryptionKeySHA256': blob._properties.get('customerEncryption', {}).get('keySha256', ''),
        }
