import os
import pyproj

from shapely import geometry
from shapely.ops import transform

class AoI:

    def __init__( self ):

        """
        placeholder
        """

        return


    def getBoundingBox(self, centroid, distance ):

        """
        placeholder
        """

        # get utm zone for centroid
        z = self.getZone(centroid )
        l = self.getLetter(centroid)

        # create projections
        self._proj = { 'geo' : pyproj.Proj(init='epsg:4326', ellps='WGS84') }

        if centroid[ 1 ] > 0:
            self._proj[ 'utm' ] =  pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
        else:
            self._proj[ 'utm' ] =  pyproj.Proj(proj='utm', zone=z, ellps='WGS84', south=True)

        # create transformations
        self._proj[ 'utm2geo' ] = pyproj.Transformer.from_proj( self._proj[ 'utm' ], self._proj[ 'geo' ] )
        self._proj[ 'geo2utm' ] = pyproj.Transformer.from_proj( self._proj[ 'geo' ], self._proj[ 'utm' ] )

        # centroid to utm
        x, y = self._proj[ 'geo2utm' ].transform(centroid[0], centroid[1])

        x0 = x - distance; y0 = y - distance
        x1 = x + distance; y1 = y + distance

        # convert back to geo
        pt0 = self._proj[ 'utm2geo' ].transform(x0, y0)
        pt1 = self._proj[ 'utm2geo' ].transform(x1, y1)

        return geometry.Polygon( [ [ pt0[ 0 ], pt0[ 1 ] ], 
                                    [ pt1[ 0 ], pt0[ 1 ] ], 
                                    [ pt1[ 0 ], pt1[ 1 ] ],
                                    [ pt0[ 0 ], pt1[ 1 ] ],
                                    [ pt0[ 0 ], pt0[ 1 ] ] ] )


    def getEpsg( self, centroid  ):

        """
        placeholder
        """

        # get utm zone for centroid
        z = self.getZone(centroid )
        l = self.getLetter(centroid)

        # create pyproj object
        proj = None
        if centroid[ 1 ] > 0:
            proj =  pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
        else:
            proj =  pyproj.Proj(proj='utm', zone=z, ellps='WGS84', south=True)

        # return epsg code
        return proj.crs.to_epsg()


    def getZone(self, lonlat):

        """
        placeholder
        """

        # decision tree for evaluate utm zone from longitude
        if 56 <= lonlat[1] < 64 and 3 <= lonlat[0] < 12:
            return 32
        if 72 <= lonlat[1] < 84 and 0 <= lonlat[0] < 42:
            if lonlat[0] < 9:
                return 31
            elif lonlat[0] < 21:
                return 33
            elif lonlat[0] < 33:
                return 35
            return 37

        return int( ( lonlat[0] + 180) / 6) + 1


    def getLetter( self, lonlat ):

        """
        placeholder
        """

        # extract designated letter code from latitude
        return 'CDEFGHJKLMNPQRSTUVWXX'[int((lonlat[1] + 80) / 8)]

"""
a=AoI()
b=a.getBoundingBox( (1,-51), 1000 )
print ( b.bounds )

c= a.toUtm( b )
print ( c.bounds )
"""

