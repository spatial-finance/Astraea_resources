import os
import numpy as np
from osgeo import gdal

class Mask:

    def __init__(self ):

        """
        Placeholder
        """

        # increase system memory usage
        os.environ['GDAL_CACHEMAX'] = '2048'
        gdal.UseExceptions()

        self._qa_vars = {
            'fill': self.fill_qa,
            'terrain': self.terrain_qa,
            'radiometricSaturation': self.radiometric_qa,
            'cloud': self.cloud,
            'cloudConf': self.cloud_confidence,
            'cirrusConf': self.cirrus_confidence,
            'cloudShadowConf': self.cloud_shadow_confidence,
            'snowIceConf': self.snow_ice_confidence,
        }

        self._binary_vars = ('terrain', 'cloud', 'fill')
        self._data = None

        return


    def generate( self, qa_pathname, out_pathname=None ):

        """
        writes the cloud+alpha mask as single-band uint8 tiff
        suitable for stacking as an alpha band
        threshold defaults to 2; only 2 and above are considered clouds
        """

        self._data = None

        # open qa file
        ds = gdal.Open( qa_pathname )
        if ds is not None:

            # load qa array into memory
            qa_data = ds.GetRasterBand(1).ReadAsArray()
            self._data = np.zeros ( qa_data.shape, dtype=int)

            # retrieve cloud confidences
            funcs = [ 'cirrusConf', 'cloudConf', 'cloudShadowConf' ]
            for func in funcs:

                # apply logical or on mask and 
                bits = self._qa_vars[ func ] ( qa_data )
                self._data = np.logical_or ( self._data, bits >= 3 )

            if out_pathname is not None:

                # get driver and create copy
                driver = gdal.GetDriverByName( 'GTiff' )
                out_ds = driver.CreateCopy( out_pathname, ds, options=[ 'TILED=YES', 'COMPRESS=DEFLATE'] )
                if out_ds is not None:

                    # write array
                    out_ds.GetRasterBand(1).WriteArray( self._data )
                    out_ds = None

            # close file
            ds = None

        return self._data


    def apply( self, pathname, out_pathname=None ):

        """
        Placeholder
        """

        channel = None

        # open dataset
        ds = gdal.Open( pathname )
        if ds is not None:

            # read data and check equivalent dimensions
            if ds.RasterYSize == self._data.shape[ 0 ] and ds.RasterXSize == self._data.shape[ 1 ]:

                # set pixels to nodata
                channel = ds.GetRasterBand(1).ReadAsArray()
                channel[ self._data > 0 ] = 0

                if out_pathname is not None:

                    # get driver and create copy
                    driver = gdal.GetDriverByName( 'GTiff' )
                    out_ds = driver.CreateCopy( out_pathname, ds, options=[ 'TILED=YES', 'COMPRESS=DEFLATE'] )
                    if out_ds is not None:

                        # write array
                        out_ds.GetRasterBand(1).WriteArray( channel )
                        out_ds.GetRasterBand(1).SetNoDataValue( 0 )

                    out_ds = None
                
            # close dataset
            ds = None
        
        return channel




    def captureBits( self, arr, b1, b2):

        """
        Placeholder
        """

        width_int = int((b1 - b2 + 1) * "1", 2)
        return ((arr >> b2) & width_int).astype('uint8')


    def fill_qa(self, arr):
        """
        0 = No, this condition does not exist
        1 = Yes, this condition exists
        """
        return self.captureBits(arr, 0, 0)


    def terrain_qa(self, arr):
        """
        0 = No, this condition does not exist
        1 = Yes, this condition exists
        """
        return self.captureBits(arr, 1, 1)


    def radiometric_qa(self, arr):
        """
        For radiometric saturation bits (2-3), read from left to right
        represent how many bands contain saturation:
        00 - No bands contain saturation
        01 - 1-2 bands contain saturation
        10 - 3-4 bands contain saturation
        11 - 5 or more bands contain saturation
        """
        return self.captureBits(arr, 3, 2)


    def cloud(self, arr):
        """
        0 = No, this condition does not exist
        1 = Yes, this condition exists
        """
        return self.captureBits(arr, 4, 4)


    def cloud_confidence(self, arr):
        """
        00 = "Not Determined" = Algorithm did not determine the status of this condition
        01 = "No" = Algorithm has low to no confidence that this condition exists (0-33 percent confidence)
        10 = "Maybe" = Algorithm has medium confidence that this condition exists (34-66 percent confidence)
        11 = "Yes" = Algorithm has high confidence that this condition exists (67-100 percent confidence
        """
        return self.captureBits(arr, 6, 5)


    def cloud_shadow_confidence(self, arr):
        """
        00 = "Not Determined" = Algorithm did not determine the status of this condition
        01 = "No" = Algorithm has low to no confidence that this condition exists (0-33 percent confidence)
        10 = "Maybe" = Algorithm has medium confidence that this condition exists (34-66 percent confidence)
        11 = "Yes" = Algorithm has high confidence that this condition exists (67-100 percent confidence
        """
        return self.captureBits(arr, 8, 7)


    def snow_ice_confidence(self, arr):
        """
        00 = "Not Determined" = Algorithm did not determine the status of this condition
        01 = "No" = Algorithm has low to no confidence that this condition exists (0-33 percent confidence)
        10 = "Maybe" = Algorithm has medium confidence that this condition exists (34-66 percent confidence)
        11 = "Yes" = Algorithm has high confidence that this condition exists (67-100 percent confidence
        """
        return self.captureBits(arr, 10, 9)


    def cirrus_confidence(self, arr):
        """
        00 = "Not Determined" = Algorithm did not determine the status of this condition
        01 = "No" = Algorithm has low to no confidence that this condition exists (0-33 percent confidence)
        10 = "Maybe" = Algorithm has medium confidence that this condition exists (34-66 percent confidence)
        11 = "Yes" = Algorithm has high confidence that this condition exists (67-100 percent confidence
        """
        return self.captureBits(arr, 12, 11)


    def lookup( self, name, val):

        """
        Placeholder
        """

        if name in self._binary_vars:
            if val == 0:
                return "no"
            return "yes"
        else:
            if val == 0:
                return "notDetermined"
            elif val == 1:
                return "no"
            elif val == 2:
                return "maybe"
            elif val == 3:
                return "yes"
