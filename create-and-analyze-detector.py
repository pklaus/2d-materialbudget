#!/usr/bin/env python

# Local dependencies
from svgtools import get_polygons, polygons_to_svg, geometry_to_svg

# External dependencies
# http://toblerity.org/shapely/manual.html
import shapely.geometry
import shapely.affinity
import shapely.speedups
# http://docs.scipy.org/doc/numpy/index.html
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# https://github.com/Infinidat/munch
from munch import munchify, Munch

# Python Stdlib
import sys
import pprint
import copy
import json
import multiprocessing
import pickle
import shutil
from datetime import datetime as dt

DEBUG = False

if shapely.speedups.available:
    shapely.speedups.enable()

class BaseComponent(object):
    """
    Representing a number of polygons sharing the same
    properties and belonging to a single layer of material budget
    """
    def __init__(self, name, polygons, material="", thickness=0.01, material_budget=0.0):
        self.name = name
        self.polygons = polygons
        self.layer = name
        self.material = material
        self.thickness = thickness
        self.material_budget = material_budget

    def contains(self, other):
        for poly in self.relevant_polygons:
            if poly.contains(other): return True
        return False

    def set_relevant_polygons(self, shape):
        len_total = len(self.polygons)
        self.relevant_polygons = [p for p in self.polygons if p.intersects(shape)]
        len_relevant = len(self.relevant_polygons)
        #print("{} component: selected {} out of {} polygons as relevant".format(self.name, len_relevant, len_total))

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

def calc_job(job):
    return job.calc()

class CalculatePatchJob(object):
    def __init__(self, patch, geometry):
        self.patch = patch
        self.geometry = geometry

    def calc(self):
        #print("calculating for position {}".format(self.patch.shape.centroid))
        r = dict() # result dict
        p = self.patch
        shape = p.shape
        bounds = shape.bounds
        bcs = self.geometry.components[self.geometry.top_level_entity].base_components
        for bc in bcs:
            bc.set_relevant_polygons(shape)
        for sample in p.samples:
            # calculate sample positions
            sp = shapely.geometry.Point(sample[0] * p.width_step + bounds[0], sample[1] * p.height_step + bounds[1])
            for bc in bcs:
                if bc.contains(sp):
                    try:
                        r[bc.material] += bc.material_budget
                    except KeyError:
                        r[bc.material] = bc.material_budget
        # normalize material budget to number of samples
        for material in r:
            r[material] = float(r[material]) / len(p.samples)
        self.result = Munch(r)
        del bcs
        del self.geometry
        return self

def parse_point(string):
    coords = string.split(',')
    return shapely.geometry.Point(float(coords[0]), float(coords[1]))

def main():
    global DEBUG

    import argparse
    parser = argparse.ArgumentParser(description="Create a detector SVG image from a geometry file stating its components")
    parser.add_argument('--debug', action='store_true', help='enable debugging output')
    parser.add_argument('--start', metavar='POINT', required=True, type=parse_point, help='Start Point')
    parser.add_argument('--end',   metavar='POINT', required=True, type=parse_point, help='End Point')
    parser.add_argument('--num-x-bins', type=int, required=True, help='Number of bins in the x direction')
    parser.add_argument('--num-y-bins', type=int, required=True, help='Number of bins in the y direction')
    parser.add_argument('--samples-per-bin', type=int, help='Number of bins in the y direction')
    parser.add_argument('--fast', action='store_true', help='Speed up imaging complicated SVGs with overlapping polygons of the same layer')
    parser.add_argument('json_geometry_file', help='The geometry file in JSON format')
    parser.add_argument('output_name', help='The basename for the output files')
    args = parser.parse_args()

    DEBUG = args.debug

    # Save the CLI --args in a file:
    args_dict = vars(args).copy()
    args_dict['start'] = str(args_dict['start'])
    args_dict['end'] = str(args_dict['end'])
    with open(args.output_name + '.args.pydict', 'w') as f:
        f.write(pprint.pformat(args_dict))
    shutil.copyfile(args.json_geometry_file, args.output_name + '.geometry.json')

    # Let's make sure that our bounds follow those rules:
    #   --start.x  <  --end.x   and   --start.y  <  --end.y
    f, t = args.start, args.end
    start = shapely.geometry.Point(min(f.x, t.x), min(f.y, t.y))
    end   = shapely.geometry.Point(max(f.x, t.x), max(f.y, t.y))
    del f, t
    print('Imaging area from {} to {}'.format(start, end))

    num_x_bins = args.num_x_bins
    num_y_bins = args.num_y_bins
    width  = end.x - start.x
    height = end.y - start.y
    width_step  = width  / num_x_bins
    height_step = height / num_y_bins

    # Starting to read in the geometry definition
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
        mb = g.materials[material].budget_per_um * thickness
        b = [BaseComponent(poly_component, polygons, material=material, thickness=thickness, material_budget=mb)]
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
            mb = g.materials[material].budget_per_um * thickness
            b = [BaseComponent(svg_component, polygons, material=material, thickness=thickness, material_budget=mb)]
            b = apply_transforms(b, cs[svg_component].transforms)
            g.components[svg_component].base_components = b
        except FileNotFoundError:
            sys.stderr.write('Could not read SVG image: ' + svg_file)

    if DEBUG: print("Calculating component tree")
    calc_base_components(g, g.top_level_entity)
    if DEBUG: print("Converting to SVG coordinate space")
    apply_transforms(g.components[g.top_level_entity].base_components, [dict(kind='scale', yfact=-1, origin=(0, 0))])
    if DEBUG: print("Creating SVG")
    dwg = geometry_to_svg(g, g.top_level_entity, g.size, [int(s/2) for s in g.size], profile='tiny')
    if DEBUG: print("Saving to SVG")
    dwg.saveas(args.output_name + '.svg')
    if DEBUG: print("Finished saving the SVG file!")


    if DEBUG: print("calculating material budget now")

    # Preparing our Patches (= bins)
    patches = dict()
    for id in range(num_x_bins * num_y_bins):
        p = Munch()
        p.id = id
        x_num = id % num_x_bins
        y_num = id // num_x_bins
        x = start.x + x_num * width_step
        y = start.y + y_num * height_step
        p.width_step = width_step
        p.height_step = height_step
        shape = shapely.geometry.box(x, y, x+width_step, y+height_step)
        p.shape = shape
        p.x_num = x_num
        p.y_num = y_num
        p.id_tuple = (x_num, y_num)
        p.samples = np.random.rand(args.samples_per_bin, 2)
        p.materials = dict()
        p.components = dict()
        patches[id] = p

    # Starting 'imaging'
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    chunksize = 3 * multiprocessing.cpu_count()
    jobs = [CalculatePatchJob(patches[id], g) for id in patches]
    #jobs = p.map(calc_job, jobs, chunksize=chunksize)
    ## instead use imap_unordered (to give more status output):
    results = []
    start_time = dt.now()
    for i, result in enumerate(p.imap_unordered(calc_job, jobs, chunksize)):
        now = dt.now()
        total_time = (now-start_time)/(i+1) * len(jobs)
        eta = start_time + total_time # estimated time of arrival
        eta = eta.replace(microsecond=0)
        print("Done with CalculatePatchJob #{} of {} ({:.1%}). ETA: {}".format(i, len(jobs), i/len(jobs), eta))
        results.append(result)
    jobs = results
    p.close()
    #for job in jobs:
    #    print(job.result)
    #import pdb; pdb.set_trace()

    # save the CalculatePatchJob results:
    for job in jobs:
        del job.patch.samples
    with open(args.output_name + '.pkl', 'wb') as f:
        pickle.dump(jobs, f, pickle.HIGHEST_PROTOCOL)

    # Reporting
    totals_layer = np.zeros((num_x_bins, num_y_bins))
    for job in jobs:
        p = job.patch
        for material in job.result:
            totals_layer[p.id_tuple] += job.result[material]
    np.save(open(args.output_name + '.npy', 'wb'), totals_layer)
    plt.imshow(totals_layer, origin="lower", interpolation="nearest", extent=[start.x, end.x, start.y, end.y])
    plt.gca().add_patch(plt.Circle((0, 0), radius=g.acceptance_radius, fill=None, edgecolor='r'))
    # create the colorbar with a better scale to the image:
    plt.colorbar(fraction=0.015, pad=0.04)
    plt.savefig(args.output_name + '.png')
    plt.savefig(args.output_name + '.eps')
    plt.show()

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()

