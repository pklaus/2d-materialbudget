#!/usr/bin/env python

"""
SVG tools
"""

# https://docs.python.org/3/library/xml.dom.minidom.html
import xml.dom.minidom
# https://pypi.python.org/pypi/svg.path
import svg.path
# http://toblerity.org/shapely/manual.html
from shapely.geometry import Polygon, Point, LineString, box
# http://svgwrite.readthedocs.org/en/latest/
import svgwrite

import re


def get_polygons(svg_filename):
    """
    Searches for polygons (and rectangles) in the SVG file
    and returns them as a list of tuples where each tuple
    stands for one polygon / rectangle and contains two entries:
      - a string with the id of the object in the svg file
      - a shapely Polygon() object created from the respective svg object
    """
    s = xml.dom.minidom.parse(svg_filename)
    polys = []
    # the width and height of the svg image (needed for our coordinate transform):
    sw = float(s.getElementsByTagName('svg')[0].getAttribute('width'))
    sh = float(s.getElementsByTagName('svg')[0].getAttribute('height'))
    for rect in s.getElementsByTagName('rect'):
        x = float(rect.getAttribute('x'))
        y = float(rect.getAttribute('y'))
        width = float(rect.getAttribute('width'))
        height = float(rect.getAttribute('height'))
        # coordinate system transform:
        y = sh - y - height
        polys.append((rect.getAttribute('id'), box(x, y, x+width, y+height)))
    for path in s.getElementsByTagName('path'):
        p = svg.path.parse_path(path.getAttribute('d'))
        if not p.closed:
            try:
                m = re.search(r'stroke-width:(\d+(\.\d+)?)', path.getAttribute('style'))
                trace_width = float(m.group(1))
                assert trace_width > 0.0
            except:
                print("Cannot handle the open path with the id {}".format(path.getAttribute('id')))
                continue
        coords = [p[0].start]
        for el in p:
            if type(el) != svg.path.Line:
                raise NameError('The SVG file contains a path with crap: {}.'.format(type(el)))
            coords.append(el.end)
        # converting the 'complex' coordinate type from svg.path.parse_path() to tuple
        # and transform coordinates from SVG to project convention at the same time
        coords = [(c.real, sh - c.imag) for c in coords]
        if not p.closed:
            ls = LineString(coords)
            polygon = ls.buffer(distance=trace_width/2.0, resolution=16, cap_style=2, join_style=1, mitre_limit=1.0)
        else:
            polygon = Polygon(coords)
        polys.append((path.getAttribute('id'), polygon))
    return polys

def polygons_to_svg(list_of_polygons, size, origin, profile='full'):
    viewbox = [-origin[0], -origin[1], size[0], size[1]]
    viewbox = ' '.join(str(el) for el in viewbox)
    dwg = svgwrite.Drawing(size=size, viewBox=viewbox, profile=profile)
    for polygon in list_of_polygons:
        dwg.add(svgwrite.shapes.Polygon(polygon, fill='red', opacity=0.5))
    return dwg

def geometry_to_svg(geometry, top_level_entity, size, origin, profile='full'):
    g = geometry
    viewbox = [-origin[0], -origin[1], size[0], size[1]]
    viewbox = ' '.join(str(el) for el in viewbox)
    dwg = svgwrite.Drawing(size=size, viewBox=viewbox, profile=profile)
    for base_component in g.components[top_level_entity].base_components:
        color = g.colorcode[base_component.material].color
        opacity = g.colorcode[base_component.material].opacity
        for polygon in base_component.polygons:
            polygon = polygon.exterior.coords
            dwg.add(svgwrite.shapes.Polygon(polygon, fill=color, opacity=opacity))
    return dwg

def parse_point(string):
    coords = string.split(',')
    return Point(float(coords[0]), float(coords[1]))

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('svg_file', help='The svg file to read in')
    parser.add_argument('points', metavar='POINT', nargs='*', type=parse_point, help='Points to check')
    parser.add_argument('--debug', action='store_true', help='Enable debugging output')
    args = parser.parse_args()
    try:
        polys = get_polygons(args.svg_file)
    except FileNotFoundError:
        parser.error('Please provide an SVG file.')
    print('{} polygons or rectangles found in the SVG file "{}".'.format(len(polys), args.svg_file))
    if args.debug:
        if polys: print("All polygons found:")
        for poly in polys:
            print(*poly)
    for p in args.points:
        print(p)
        for poly in polys:
            if poly[1].contains(p): print(" ↳ hits {}".format(poly[0]))
            #else: print(p, poly[1])

if __name__ == "__main__":
    main()

