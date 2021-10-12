import os
import glob
from osgeo import gdal
from scraper import TileScraper
from tiler import MercatorTiler, SlippyTiler

class Downloader:

    def __init__( self, url, options=None, img_format='png', profile='slippy', threads=6 ):

        """
        constructor
        """

        # copy arguments    
        self._url = url
        self._format = img_format
        self._options = options
        self._threads = threads

        # create flavour of tiler object
        if profile == 'mercator':
            self._tiler = MercatorTiler()
        else:
            self._tiler = SlippyTiler()

        return


    def process ( self, bbox, zoom, out_pathname ):

        """
        placeholder
        """

        # get tile x, y coordinates of bbox
        x1, y1 = self._tiler.LatLonToTile( bbox[ 1 ], bbox[ 0 ], zoom )
        x2, y2 = self._tiler.LatLonToTile( bbox[ 3 ], bbox[ 2 ], zoom )

        # create output folder if required
        out_path = os.path.dirname( out_pathname )
        if not os.path.exists( out_path ):
            os.makedirs( out_path )

        # get tasklist for threads
        threads = []
        tasklist = self.getTaskList( (x1,y1,x2,y2), zoom, out_path )
        
        for idx, tasks in enumerate( tasklist ):

            # create derived tilescraper object 
            obj = TileScraper( idx, tasks, { 'tiler' : self._tiler, 'options' : self._options } )
            obj.start()
            threads.append( obj )

        # pause main thread until all child threads complete
        for obj in threads:
            obj.join()

        # generate list of newly downloaded tiles
        tiles = []
        for thread in threads:
            tiles.extend ( thread._tiles )

        # merge tiles into single image
        gdal.Warp( out_pathname, tiles )        
        if os.path.exists( out_pathname ):

            # setup copy
            driver = gdal.GetDriverByName('GTiff')
            copy_pathname = out_pathname.replace( '.tif', '-copy.tif' )

            # apply compression and tiling
            ds = gdal.Open( out_pathname, gdal.GA_ReadOnly )
            ds_copy = driver.CreateCopy( copy_pathname, ds, options=[ "TILED=YES", "COMPRESS=LZW" ] )
            ds = None; ds_copy = None

            # apply compression and tiling
            os.remove( out_pathname )
            os.rename( copy_pathname, out_pathname )

        # remove tile files
        files = glob.glob( os.path.join( out_path, 'tile*' ) )
        for f in files:
            os.remove( f )

        # remove path if empty
        os.rmdir( out_path )

        return


    def getTaskList( self, bbox, zoom, out_path ):

        """
        get tile image from https url
        """

        # initialise list of tasklists
        tasklist = [ [] for t in range( self._threads ) ]
        x1, y1, x2, y2 = bbox

        # loop through tile coordinates
        index = 0
        for y in range( min( y1, y2 ), max( y1, y2 ) + 1 ):
            for x in range( x1, x2 + 1 ):

                # construct url
                values = { 'x' : x, 'y' : y, 'z' : zoom, 'format' : self._format }
                url = self._url.format( **values ) 

                # download tile to pathname
                pathname = os.path.join( out_path, 'tile_{}_{}_{}.{}'.format( zoom, x, y, self._format ) ) 

                # append job to list
                tasklist[ index ].append( { 'xyz' : ( x, y, zoom ), 'url' : url, 'pathname' : pathname } )
                index += 1
                
                # reset thread / taskset index
                if index == self._threads:
                    index = 0

        return tasklist

