#!/usr/bin/env python

# Local dependencies
#from create-and-analyze-detector import BaseComponent, CalculatePatchJob

# External dependencies
# http://docs.scipy.org/doc/numpy/index.html
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
# https://github.com/Infinidat/munch
from munch import munchify, Munch

# Python Stdlib
import pprint
import copy
import json
import multiprocessing
import pickle
import math

DEBUG = False

def bin_idx(pos, bin_limits):
    #if pos < bin_start or pos > bin_end:
    if pos < bin_limits[0] or pos > bin_limits[-1]:
        return None
    i = 0
    while pos > bin_limits[i+1]:
        i += 1
    return i

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

class CalculatePatchJob(object):
    def __init__(self, patch, geometry):
        self.patch = patch
        self.geometry = geometry

    def calc(self):
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
        return self

def main():
    global DEBUG

    import argparse
    parser = argparse.ArgumentParser(description="Circular-calc")
    parser.add_argument('--debug', action='store_true', help='enable debugging output')
    parser.add_argument('input_file', help='The input file')
    parser.add_argument('output_name', help='The basename for the output files')
    args = parser.parse_args()

    DEBUG = args.debug


    with open(args.input_file, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        jobs = pickle.load(f)

    num_bins = 62
    bin_start = 0
    bin_end = 30
    angles = np.linspace(bin_start, bin_end, num=(num_bins+1))
    values = dict()
    hits = np.zeros(num_bins)
    for job in jobs:
        pos = job.patch.shape.centroid
        dist = (pos.x**2+pos.y**2)**(.5)
        theta = math.atan(dist/150.)*180/math.pi
        idx = bin_idx(theta, angles)
        if idx is None: continue
        for material in job.result:
            try:
                values[material]
            except KeyError:
                values[material] = np.zeros(num_bins)
            values[material][idx] += job.result[material]/math.cos(theta*math.pi/180.)
        hits[idx] += 1
    bottom = np.zeros(num_bins)
    color_index = 0
    colors = ['#727272', '#f1595f', '#885555', '#f9a65a', '#b87333', '#599ad3', '#79c36a', '#9e66ab', '#cd7058', '#d77fb3', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plots = dict()
    materials = [material for material in values]
    #materials = ['TPG', 'glue', 'polyimide', 'Cu', 'Si']
    materials = ['TPG', 'glue', 'Si', 'polyimide', 'Cu']
    widths = angles[1:] - angles[:-1]
    angles = angles[:-1]
    for material in materials:
        values[material] = values[material]/hits
        plots[material] = plt.bar(angles, values[material], width=widths, bottom=bottom, color=colors[color_index])
        bottom += values[material]
        color_index += 1
        color_index %= 9
    plt.ylabel(r'Material budget $x/X_0 [\%]$')
    plt.xlabel(r'Polar angle $\vartheta$ $[^\circ]$')
    plt.title('3rd MVD station')
    #plt.text(r'integrated over the azimuthal angle $\varphi$')
    lrefs = [plots[material][0] for material in materials]
    plt.legend(reversed(lrefs), reversed(materials), loc=2)
    plt.gca().add_line(plt.Line2D((2.5, 25), (0.5, 0.5), lw=2, color='r'))
    plt.gca().add_line(plt.Line2D((25, 25), (0, 0.5), lw=2, color='r'))
    plt.gca().add_line(plt.Line2D((2.5, 2.5), (0, 0.5), lw=2, color='r'))
    plt.savefig(args.output_name + '.theta.png')
    plt.savefig(args.output_name + '.theta.eps')
    plt.show()

    #import pdb; pdb.set_trace()
    num_x_bins, num_y_bins = 0, 0
    for job in jobs:
        num_x_bins = max(num_x_bins, job.patch.x_num)
        num_y_bins = max(num_y_bins, job.patch.y_num)
    num_x_bins += 1; num_y_bins += 1
    totals_layer = np.zeros((num_x_bins, num_y_bins))
    for job in jobs:
        p = job.patch
        for material in job.result:
            totals_layer[p.id_tuple] += job.result[material]
    cmap = matplotlib.colors.LinearSegmentedColormap(
    'Custom Color Map',
    # original jet from https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/_cm.py#L260
    #segmentdata = {
    #    'red':   ( (0.,  .0,  .0), (0.35,  0, 0), (0.66,  1, 1), (0.89, 1, 1), (1,    0.5, 0.5)),
    #    'green': ( (0.,  .0,  .0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91,  .0,  .0), (1, 0, 0)),
    #    'blue':  ( (0., 0.5, 0.5), (0.11,  1, 1), (0.34,  1, 1), (0.65, 0, 0), (1,     .0,  .0))
    #}
    segmentdata = {
        'red':   ( (0.,  .0,  .0), (0.39,   0, 0), (0.55,   0.6, 0.6), (0.69,   0, 0), (0.9,  1, 1), (0.95, 1, 1), (1,    0.5, 0.5)),
        'green': ( (0.,  .0,  .0), (0.175,  0, 0), (0.45, 1, 1), (0.75, 1, 1), (0.96,  .0,  .0), (1, 0, 0)),
        'blue':  ( (0., 0.5, 0.5), (0.154,  1, 1), (0.45, 1, 1), (0.75, 0, 0), (1,     .0,  .0))
    }
    #segmentdata = {
    #     'red':   [(0.0,  0.0, 0.0),
    #               (0.3,  0.1, 0.1),
    #               (0.7,  0.6, 0.6),
    #               (0.85,  0.1, 0.1),
    #               (0.94,  0.0, 0.0),
    #               (1.0,  1.0, 1.0)],

    #     'green': [(0.0,  0.0, 0.0),
    #               (0.2,  0.2, 0.2),
    #               (0.4, 0.6, 0.6),
    #               (0.8, .0, .0),
    #               (0.95, 1.0, 1.0),
    #               (1.0,  0.0, 0.0)],

    #     'blue':  [(0.0,  0.2, 0.2),
    #               (0.1,  1.0, 1.0),
    #               (0.25,  0.0, 0.0),
    #               (1.0,  0.0, 0.0)]}
    )
    norm = matplotlib.colors.Normalize()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.67)
    plt.imshow(totals_layer, cmap=cmap, norm=norm, aspect='equal', origin="lower", interpolation="nearest", extent=[-80, 80, -80, 80])
    plt.gca().add_patch(plt.Circle((0, 0), radius=69.946, fill=None, edgecolor='r'))
    # create the colorbar with a better scale to the image:
    plt.colorbar(fraction=0.015, pad=0.04)
    plt.savefig(args.output_name + '.png')
    plt.savefig(args.output_name + '.eps')
    plt.show()


if __name__ == "__main__":
    main()

