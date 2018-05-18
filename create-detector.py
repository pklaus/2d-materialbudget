#!/usr/bin/env python

# Local dependencies
from svgtools import get_polygons, polygons_to_svg, geometry_to_svg

# External dependencies
# http://toblerity.org/shapely/manual.html
import shapely.geometry
import shapely.affinity
# http://docs.scipy.org/doc/numpy/index.html
import numpy as np
from matplotlib import pyplot as plt
# https://github.com/Infinidat/munch
from munch import munchify, Munch

# Python Stdlib
import pprint
import copy
import json

DEBUG = False

class BaseComponent(object):
    """
    Representing a number of polygons sharing the same
    properties and belonging to a single layer of material budget
    """
    def __init__(self, name, polygons, material="", thickness=0.01):
        self.name = name
        self.polygons = polygons
        self.layer = name
        self.material = material
        self.thickness = thickness

def apply_transforms(base_components, transforms):
    for transform in transforms:
        if DEBUG: print('Performing a transform of kind "{kind}"'.format(**transform))
        t = dict(xfact=1.0, yfact=1.0, xoff=0.0, yoff=0.0, origin='center')
        t.update(dict(transform))
        t = Munch(t)
        for base_component in base_components:
            #import pdb; pdb.set_trace()
            polygons = base_component.polygons
            if t.kind == 'scale':
                t.origin = tuple(t.origin)
                polygons = [shapely.affinity.scale(p, xfact=t.xfact, yfact=t.yfact, origin=t.origin) for p in polygons]
            elif t.kind == 'translate':
                polygons = [shapely.affinity.translate(p, xoff=t.xoff, yoff=t.yoff) for p in polygons]
            elif t.kind == 'rotate':
                polygons = [shapely.affinity.rotate(p, angle=t.angle, origin=t.origin) for p in polygons]
            
            base_component.polygons = polygons
    return base_components

def apply_transforms_polygons(polygons, transforms):
    for transform in transforms:
        if DEBUG: print('Performing a transform of kind "{kind}"'.format(**transform))
        t = dict(xfact=1.0, yfact=1.0, xoff=0.0, yoff=0.0, origin='center')
        t.update(dict(transform))
        t = Munch(t)
        if t.kind == 'scale':
            t.origin = tuple(t.origin)
            polygons = [shapely.affinity.scale(p, xfact=t.xfact, yfact=t.yfact, origin=t.origin) for p in polygons]
        elif t.kind == 'translate':
            polygons = [shapely.affinity.translate(p, xoff=t.xoff, yoff=t.yoff) for p in polygons]
        elif t.kind == 'rotate':
            polygons = [shapely.affinity.rotate(p, angle=t.angle, origin=t.origin) for p in polygons]
    return polygons

def calc_base_components(g, entity):
    try:
        return g.components[entity].base_components
    except:
        base_components = []
        for comp in g.components[entity].components:
            b = calc_base_components(g, comp.name)
            b = [copy.copy(bc) for bc in b]
            apply_transforms(b, comp.transforms)
            #import pdb; pdb.set_trace()
            base_components += b
        g.components[entity].base_components = base_components
        return g.components[entity].base_components


def calc_polygons(g, entity):
    try:
        return g.components[entity].polygons
    except:
        polygons = []
        for comp in g.components[entity].components:
            polys = calc_polygons(g, comp.name)
            polys = apply_transforms_polygons(polys, comp.transforms)
            polygons += polys
        g.components[entity].polygons = polygons
        return g.components[entity].polygons

def main():
    global DEBUG

    import argparse
    parser = argparse.ArgumentParser(description="Create a detector SVG image from a geometry file stating its components")
    parser.add_argument('--debug', action='store_true', help='enable debugging output')
    parser.add_argument('json_geometry_file', help='The geometry file in JSON format')
    parser.add_argument('output_svg', help='The name of the output file in SVG format')
    args = parser.parse_args()

    DEBUG = args.debug

    with open(args.json_geometry_file, 'r') as geometry_file:
        g = json.load(geometry_file)
    g = munchify(g)

    cs = g.components

    # Set .base_components for the components of kind == polygon
    poly_components = [c for c in cs if cs[c].kind == 'polygon']
    for poly_component in poly_components:
        if DEBUG: print("Reading polygon {}".format(poly_component))
        polygons = [shapely.geometry.Polygon(cs[poly_component].points)]
        material = cs[poly_component].material
        thickness = cs[poly_component].thickness
        b = [BaseComponent(poly_component, polygons, material=material, thickness=thickness)]
        b = apply_transforms(b, cs[poly_component].transforms)
        g.components[poly_component].base_components = b

    # Set .base_components for the components of kind == svg
    svg_components = [c for c in cs if cs[c].kind == 'svg']
    for svg_component in svg_components:
        svg_file = cs[svg_component].filename
        try:
            polygons = get_polygons(svg_file)
            polygons = [polygon[1] for polygon in polygons]
            material = cs[svg_component].material
            thickness = cs[svg_component].thickness
            b = [BaseComponent(svg_component, polygons, material=material, thickness=thickness)]
            b = apply_transforms(b, cs[svg_component].transforms)
            g.components[svg_component].base_components = b
        except FileNotFoundError:
            sys.stderr.write('Could not read SVG image: ' + svg_file)

    # polygons = calc_polygons(g, g.top_level_entity)
    # # Convert back to svg coordinates:
    # polygons = apply_transforms(polygons, [dict(kind='scale', yfact=-1, origin=(0, 0))])
    # # Convert to svgwrite drawing:
    # dwg = polygons_to_svg(polygons, g.size, [int(s/2) for s in g.size])

    if DEBUG: print("Calculating component tree")
    calc_base_components(g, g.top_level_entity)
    if DEBUG: print("Converting to SVG coordinate space")
    apply_transforms(g.components[g.top_level_entity].base_components, [dict(kind='scale', yfact=-1, origin=(0, 0))])
    if DEBUG: print("Creating SVG")
    dwg = geometry_to_svg(g, g.top_level_entity, g.size, [int(s/2) for s in g.size])
    if DEBUG: print("Saving to SVG")
    #with open(args.output_svg, 'w') as outfile:
    #    outfile.write(dwg.tostring())
    dwg.saveas(args.output_svg)
    if DEBUG: print("Finished saving the SVG file!")


if __name__ == "__main__":
    main()

