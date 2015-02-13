#!/usr/bin/env python

"""
Imaging SVG images.

This tool 'images' SVG images. By imaging we mean binning
the vector image and analyzing each bin with Monte Carlo sampling.
"""

# http://toblerity.org/shapely/manual.html
from shapely.geometry import Polygon, Point, box
# http://docs.scipy.org/doc/numpy/index.html
import numpy as np
from matplotlib import pyplot as plt

from svgtools import get_polygons

def parse_point(string):
    coords = string.split(',')
    return Point(float(coords[0]), float(coords[1]))

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', metavar='POINT', required=True, type=parse_point, help='Start Point')
    parser.add_argument('--end',   metavar='POINT', required=True, type=parse_point, help='End Point')
    parser.add_argument('--num-x-bins', type=int, required=True, help='Number of bins in the x direction')
    parser.add_argument('--num-y-bins', type=int, required=True, help='Number of bins in the y direction')
    parser.add_argument('--samples-per-bin', type=float, help='Number of bins in the y direction')
    parser.add_argument('--debug', action='store_true', help='enable debugging output')
    parser.add_argument('svg_file', help='The svg file to read in')
    args = parser.parse_args()
    try:
        polys = get_polygons(args.svg_file)
    except FileNotFoundError:
        parser.error('Please provide an SVG file.')

    # Let's make sure that our bounds follow those rules:
    #   --start.x  <  --end.x   and   --start.y  <  --end.y
    f, t = args.start, args.end
    start = Point(min(f.x, t.x), min(f.y, t.y))
    end   = Point(max(f.x, t.x), max(f.y, t.y))
    del f, t
    print('Imaging area from {} to {}'.format(start, end))

    num_x_bins = args.num_x_bins
    num_y_bins = args.num_y_bins
    width  = end.x - start.x
    height = end.y - start.y
    width_step  = width  / num_x_bins
    height_step = height / num_y_bins
    # Preparing our bins
    bins = dict()
    for id in range(num_x_bins * num_y_bins):
        b = dict()
        x_num = id % num_x_bins
        y_num = id // num_x_bins
        x = start.x + x_num * width_step
        y = start.y + y_num * height_step
        b['id'] = id
        b['x_num'] = x_num
        b['y_num'] = y_num
        b['id_tuple'] = (x_num, y_num)
        b['shape'] = box(x, y, x+width_step, y+height_step)
        b['samples'] = np.random.rand(args.samples_per_bin, 2)
        b['layers'] = dict()
        bins[id] = b
    # Starting 'imaging'
    print('{} polygons, lines (with a non-0 line-width) or rectangles found in the SVG file "{}".'.format(len(polys), args.svg_file))
    for id in bins:
        b = bins[id]
        bounds = b['shape'].bounds
        for sample in b['samples']:
            # calculate sample positions
            sp = Point(sample[0] * width_step + bounds[0], sample[1] * height_step + bounds[1])
            for poly in polys:
                if poly[1].contains(sp):
                    try:
                        b['layers'][poly[0]]['total_hits'] += 1
                    except KeyError:
                        b['layers'][poly[0]] = dict(total_hits=1)
        # calculate percentage of hits for each layer
        for layer in b['layers']:
            b['layers'][layer]['norm_hits'] = float(b['layers'][layer]['total_hits']) / args.samples_per_bin
        bins[id] = b
    # Reporting
    per_layer_maps = dict()
    totals_layer = np.zeros((num_x_bins, num_y_bins))
    for id in bins:
        b = bins[id]
        if args.debug: print("Bin {} - {}".format(id, b['shape']))
        for layer in b['layers']:
            if args.debug: print("Layer {} hit {:.2f} %".format(layer, b['layers'][layer]['norm_hits']*100))
            try:
                per_layer_maps[layer]
            except KeyError:
                per_layer_maps[layer] = np.zeros((num_x_bins, num_y_bins))
            per_layer_maps[layer][b['id_tuple']] = b['layers'][layer]['norm_hits']*100
    for layer in per_layer_maps:
        totals_layer += per_layer_maps[layer]
    np.save(open('resulting-material-budget.npy', 'wb'), totals_layer)
    print("Showing totals layer")
    plt.imshow(totals_layer.T, origin="lower", interpolation="nearest", extent=[start.x, end.x, start.y, end.y])
    plt.colorbar()
    plt.savefig('resulting-material-budget.eps')
    plt.show()
    import pdb; pdb.set_trace()
    #for layer in per_layer_maps:
    #    print("Showing plot for layer {}".format(layer))
    #    #plt.imshow(per_layer_maps[layer], origin="lower", interpolation="gaussian",extent=[start.x, end.x, start.y, end.y])
    #    plt.imshow(per_layer_maps[layer].T, origin="lower", extent=[start.x, end.x, start.y, end.y])
    #    plt.show()

if __name__ == "__main__":
    main()

