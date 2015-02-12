#!/usr/bin/env python

# https://docs.python.org/3/library/xml.dom.minidom.html
import xml.dom.minidom
# https://pypi.python.org/pypi/svg.path
import svg.path
# http://toblerity.org/shapely/manual.html
from shapely.geometry import Polygon, Point


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
        coords = [(x, y), (x+width, y), (x+width, y+height), (x, y+height), (x, y)]
        polys.append((rect.getAttribute('id'), Polygon(coords)))
    for path in s.getElementsByTagName('path'):
        p = svg.path.parse_path(path.getAttribute('d'))
        if not p.closed: continue
        coords = [p[0].start]
        for el in p:
            if type(el) != svg.path.Line:
                raise NameError('The SVG file contains a path with crap: {}.'.format(type(el)))
            coords.append(el.end)
        coords = [(c.real, c.imag) for c in coords]
        polys.append((path.getAttribute('id'), Polygon(coords)))
    return polys


def parse_point(string):
    coords = string.split(',')
    return Point(float(coords[0]), float(coords[1]))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Testing svgtools.')
    parser.add_argument('svg_file', help='The svg file to read in')
    parser.add_argument('points', metavar='POINT', nargs='*', type=parse_point, help='Points to check')
    args = parser.parse_args()
    try:
        polys = get_polygons(args.svg_file)
    except FileNotFoundError:
        parser.error('Please provide an SVG file.')
    print('{} polygons or rectangles found in the SVG file "{}".'.format(len(polys), args.svg_file))
    for p in args.points:
        print(p)
        for poly in polys:
            if poly[1].contains(p): print(" ↳ hits {}".format(poly[0]))
            #else: print(p, poly[1])

if __name__ == "__main__":
    main()

