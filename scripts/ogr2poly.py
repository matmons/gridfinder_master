#!/usr/bin/env python

# Update: A newer version can be found in the OSM SVN:
# http://trac.openstreetmap.org/browser/applications/utils/osm-extract/polygons/ogr2poly.py

# This converts OGR supported files (Shapefile, GPX, etc.) to the polygon
# filter file format [1] supported by Osmosis and other tools. It also
# includes buffering and simplifying. This allows point or line features
# to be used when creating POLY files, but in this case buffering must
# be used.
# 
# [1] http://wiki.openstreetmap.org/wiki/Osmosis/Polygon_Filter_File_Format

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
import sys
import os
from optparse import OptionParser

from osgeo import ogr
from osgeo import osr

# TODO:
#  check if file exists, make sure field is unique (increment)
#  likely doesn't handle areas spanning the antimeridian (+/-180 degrees longitude)

def createPolys(inOgr, options):
    print("Opening datasource '%s'" % inOgr)
    ds = ogr.Open(inOgr)
    lyr = ds.GetLayer(options.layer)
    
    # create SRS transformations
    mercSRS = osr.SpatialReference()
    mercSRS.ImportFromEPSG(3857) # TODO: make this an option
    wgsSRS = osr.SpatialReference()
    wgsSRS.ImportFromEPSG(4326)
    nativeSRS2bufferSRS = osr.CoordinateTransformation(lyr.GetSpatialRef(), mercSRS)
    bufferSRS2wgsSRS = osr.CoordinateTransformation(mercSRS, wgsSRS)
    nativeSRS2wgsSRS = osr.CoordinateTransformation(lyr.GetSpatialRef(), wgsSRS)

    # if no field name provided, use incrementing number (padded with just enough zeros)
    inc = 0
    incFmt = '%0' + str(len(str(lyr.GetFeatureCount()-1))) + 'd'
    
    print('Found %d features, will try and create one POLY file for each one' % lyr.GetFeatureCount())
    
    # create POLYs
    for feat in lyr:
        if options.fieldName != None:
            fieldVal = feat.GetFieldAsString(options.fieldName)
            if fieldVal is None:
                return False
            polyName = options.outPrefix + fieldVal.replace(' ', '_')
        else:
            polyName = options.outPrefix + incFmt % inc
            inc += 1
        
        print('Creating ' + polyName + '.poly')
        f = open(polyName + '.poly', 'wt')
        print(f, polyName)
        
        # this will be a polygon, TODO: handle linestrings (must be buffered)
        geom = feat.GetGeometryRef()
        geomType = geom.GetGeometryType()

        subGeom = []

        nonAreaTypes = [ogr.wkbPoint, ogr.wkbLineString, ogr.wkbMultiPoint, ogr.wkbMultiLineString]
        if geomType in nonAreaTypes and float(options.bufferDistance) == 0:
            print("Warning: Ignoring non-area type. To include you must set a buffer distance.")
            continue
        if geomType in [ogr.wkbUnknown, ogr.wkbNone]:
            print("Warning: Ignoring unknown geometry type.")
            continue

        # transform to WGS84, buffering/simplifying along the way
        if float(options.bufferDistance) > 0 or float(options.simplifyDistance) > 0:
            geom.Transform(nativeSRS2bufferSRS)
            
            if float(options.bufferDistance) > 0:
                geom = geom.Buffer(float(options.bufferDistance))
            if float(options.simplifyDistance) > 0:
                geom = geom.Simplify(float(options.simplifyDistance))
            
            geom.Transform(bufferSRS2wgsSRS)
        else:
            geom.Transform(nativeSRS2wgsSRS)

        print("dim %s, type %s" % (geom.GetDimension(), geom.GetGeometryType()))
          
        # handle multi-polygons
        subgeom = []
        geomtype = geom.GetGeometryType()
        if geomtype == ogr.wkbPolygon:
            subgeom = [geom]
        elif geomtype == ogr.wkbMultiPolygon:
            for k in range(geom.GetGeometryCount()):
                subgeom.append(geom.GetGeometryRef(k))

        print( "# of polygons: " + str(len(subgeom)))
        for g in subgeom:
            # loop over all rings in the polygon
            print( '# of rings: ' + str(g.GetGeometryCount()))
            for i in range(0, g.GetGeometryCount()):
                if i == 0:
                    # outer ring
                    print( f, i+1)
                else:
                    # inner ring
                    print( f, '!%d' % (i+1))
                ring = g.GetGeometryRef(i)
                # output all points in the ring
                print( '# of points: ' + str(ring.GetPointCount()))
                for j in range(0, ring.GetPointCount()):
                    (x, y, z) = ring.GetPoint(j)
                    print( f, '   %.6E   %.6E' % (x, y))
                print( f, 'END')
        print( f, 'END')
        f.close()
    return True

if __name__=='__main__':
    # Setup program usage
    usage = "Usage: %prog [options] src_datasource_name [layer]"
    parser = OptionParser(usage=usage)
    parser.add_option("-p", "--prefix", dest="outPrefix",
                      help="Text to prepend to POLY filenames.")
    parser.add_option("-b", "--buffer-distance", dest="bufferDistance",
                      help="Set buffer distance in meters (default: 0).")
    parser.add_option("-s", "--simplify-distance", dest="simplifyDistance",
                      help="Set simplify tolerance in meters (default: 0).")
    parser.add_option("-f", "--field-name", dest="fieldName",
                      help="Field name to use to name files.")

    parser.set_defaults(bufferDistance=0, fieldName=None, outPrefix=None,
        simplifyDistance=0, layer=0)

    # Parse and process arguments
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.print_help()
        parser.error("Error: you must specify an OGR source")
        sys.exit(1)
    elif len(args) > 2:
        parser.error("Error: you have specified too many arguments")
    
    # note that this may be a file (e.g. .shp) or a database connection string
    src_datasource = args[0]
    if len(args) == 2:
        options.layer = args[1]

    # check options
    if options.outPrefix == None:
        if os.path.exists(src_datasource):
            # put in current dir, TODO: allow user to specify output dir?
            (options.outPrefix, ext) = os.path.splitext(os.path.basename(src_datasource))
            options.outPrefix += '_'
        else:
            # file doesn't exist, so possibly a DB connection string
            options.outPrefix = 'poly_'
    if float(options.bufferDistance) < 0:
        parser.error("Buffer distance must be greater than zero.")
    if float(options.simplifyDistance) < 0:
        parser.error("Simplify tolerance must be greater than zero.")
    if float(options.simplifyDistance) > float(options.bufferDistance):
        print( "Warning: simplify distance greater than buffer distance")

        
    if createPolys(src_datasource, options):
        print( 'Finished!')
        sys.exit(0)
    else:
        print( 'Failed!')
        sys.exit(1)