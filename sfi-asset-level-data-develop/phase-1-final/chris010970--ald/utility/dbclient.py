import sys
import asyncio
import asyncpg
import pandas as pd
import geopandas as gpd
import shapely
from shapely import wkt

import nest_asyncio
nest_asyncio.apply()

class DbClient:

    # initialisation
    def __init__(self, **kwargs ):

        """
        Placeholder
        """

        obj = kwargs.pop( 'client', None )
        if obj is not None:

            # copy from existing object
            self._user = obj._user
            self._host = obj._host
            self._database = obj._database

        else:

            # pop from arguments
            self._user = kwargs.pop( 'user', None ) 
            self._host = kwargs.pop( 'host', None )
            self._database = kwargs.pop( 'database', None )

        return


    def executeQuery( self, query ):

        """
        Placeholder
        """
           
        async def execute ( query ):

            """
            Placeholder
            """

            # connect to database 
            conn = await asyncpg.connect( user=self._user, database=self._database, host=self._host )
            try:

                await conn.execute( query )

            finally:
                # close connection
                await conn.close()

            return 

        return asyncio.get_event_loop().run_until_complete( execute( query ) )


    def fetchAsDataFrame( self, query, **kwargs ):

        """
        Placeholder
        """
           
        async def execute ( query, **kwargs ):

            """
            Placeholder
            """

            # connect to database 
            conn = await asyncpg.connect( user=self._user, database=self._database, host=self._host )
            records = None
            try:

                # get column info
                stmt = await conn.prepare( query )

                # fetch records
                columns = [a.name for a in stmt.get_attributes()]
                records = await stmt.fetch()

            finally:
                # close connection
                await conn.close()

            # parse arguments for data frame construction
            coerce_float = kwargs.pop( 'coerce_float', True ) 
            parse_dates = kwargs.pop( 'parse_dates', None )
            index_col = kwargs.pop( 'index_col', None )
            params = kwargs.pop( 'params', None )

            return pd.DataFrame( records, columns=columns )

        return asyncio.get_event_loop().run_until_complete( execute( query, **kwargs ) )


    def fetchAsGeoFrame( self, query, **kwargs ):

        """
        Placeholder
        """
           
        # get frame
        df = self.fetchAsDataFrame( query, **kwargs )
        geom_col = kwargs.pop( 'geom_col', 'geom' )

        if geom_col not in df:
            raise ValueError("Query missing geometry column '{}'".format(geom_col))

        # drop nan values from geom column
        geoms = df[geom_col].dropna()
        if not geoms.empty:

            df[geom_col] = geoms = df[ geom_col ].apply(shapely.wkt.loads)

            # determine coordinate reference system
            crs = kwargs.pop( 'crs', None )
            if crs is None:

                # if no defined SRID in geodatabase, returns SRID of 0
                srid = shapely.geos.lgeos.GEOSGetSRID(geoms.iat[0]._geom)
                if srid != 0:
                    crs = {"init": "epsg:{}".format(srid)}

        return gpd.GeoDataFrame(df, crs=crs, geometry=geom_col)


    def fetchRecords( self, query ):

        """
        Placeholder
        """
           
        async def execute ( query ):

            """
            Placeholder
            """

            # connect to database 
            conn = await asyncpg.connect( user=self._user, database=self._database, host=self._host )
            records = None
            try:

                # get column info
                records = await conn.fetch( query )

            finally:
                # close connection
                await conn.close()

            return records

        return asyncio.get_event_loop().run_until_complete( execute( query ) )


    def fetchRecord( self, query ):

        """
        Placeholder
        """
           
        async def execute ( query ):

            """
            Placeholder
            """

            # connect to database 
            conn = await asyncpg.connect( user=self._user, database=self._database, host=self._host )
            record = None
            try:

                # get column info
                record = await conn.fetchrow( query )

            finally:
                # close connection
                await conn.close()

            return record

        return asyncio.get_event_loop().run_until_complete( execute( query ) )

