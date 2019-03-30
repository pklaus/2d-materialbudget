#!/usr/bin/env python

"""
This tool 'images' SVG images.

By imaging we mean defining a meshgrid on top of a vector image
describing the geometry and probing the material budget in each
grid cell / bin with with Monte Carlo sampling.
"""

# http://toblerity.org/shapely/manual.html
import shapely.geometry
# http://docs.scipy.org/doc/numpy/index.html
import numpy as np
from matplotlib import pyplot as plt
import pprint

from svgtools import get_polygons

def parse_point(string):
    coords = string.split(',')
    return shapely.geometry.Point(float(coords[0]), float(coords[1]))



class Shape(object):
    """
    Representing a (2d projection of a) geometrical shape
    with additional properties reflecting its material budget.
    """
    def __init__(self, name, polygon, layer="", material="", thickness=0.01, component=""):
        self.name = name
        self.polygon = polygon
        self.layer = name
        self.material = material
        self.component = component
        self.thickness = thickness

def Contour(object):
    """
    Contains all Shapes making up a single layer of material.
    """
    def __init__(self, name, shapes, material="", thickness=0.01, component=""):
        self.shapes = shapes
        #for shape in shapes:
        #    assert shape.material = material

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', metavar='POINT', required=True, type=parse_point, help='Start Point')
    parser.add_argument('--end',   metavar='POINT', required=True, type=parse_point, help='End Point')
    parser.add_argument('--num-x-bins', type=int, required=True, help='Number of bins in the x direction')
    parser.add_argument('--num-y-bins', type=int, required=True, help='Number of bins in the y direction')
    parser.add_argument('--samples-per-bin', type=float, help='Number of bins in the y direction')
    parser.add_argument('--debug', action='store_true', help='enable debugging output')
    parser.add_argument('--fast', action='store_true', help='Speed up imaging complicated SVGs with overlapping polygons of the same layer')
    parser.add_argument('svg_file', help='The svg file to read in')
    parser.add_argument('output_name', nargs='?', help='The name of your output files', default='resulting-material-budget')
    args = parser.parse_args()
    try:
        polys = get_polygons(args.svg_file)
    except FileNotFoundError:
        parser.error('Please provide an SVG file.')

    # Save the CLI --args in a file:
    args_dict = vars(args).copy()
    args_dict['start'] = str(args_dict['start'])
    args_dict['end'] = str(args_dict['end'])
    with open(args.output_name + '.args.pydict', 'w') as f: f.write(pprint.pformat(args_dict))

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

    # Preparing our Patches / bins
    patches = dict()
    for id in range(num_x_bins * num_y_bins):
        shape = shapely.geometry.box(x, y, x+width_step, y+height_step)
        p = Munch()
        p.id = id
        p.shape = shape
        x_num = id % num_x_bins
        y_num = id // num_x_bins
        x = start.x + x_num * width_step
        y = start.y + y_num * height_step
        p.x_num = x_num
        p.y_num = y_num
        p.id_tuple = (x_num, y_num)
        p.samples = np.random.rand(args.samples_per_bin, 2)
        p.materials = dict()
        p.components = dict()
        patches[id] = p

    # Starting 'imaging'
    print('{} polygons, lines (with a non-0 line-width) or rectangles found in the SVG file "{}".'.format(len(polys), args.svg_file))
    for id in patches:
        p = patches[id]

        bounds = b['shape'].bounds
        shape = b['shape']
        potential_polys = [poly for poly in polys if poly[1].intersects(shape)]
        if len(potential_polys): print("Number of potential polygons: {}".format(len(potential_polys)))
        for sample in b['samples']:
            # calculate sample positions
            sp = shapely.geometry.Point(sample[0] * width_step + bounds[0], sample[1] * height_step + bounds[1])
            if bl_poly[1].contains(sp): b['mb_layers']['baselayer']['total_hits'] += 1
            other_layer_hit = False
            for poly in potential_polys:
                if poly[1].contains(sp):
                    if not poly[0] == 'baselayer':
                        other_layer_hit = True
                        b['mb_layers']['other_layers']['total_hits'] += 1
                        if args.fast: break
                    try:
                        b['layers'][poly[0]]['total_hits'] += 1
                    except KeyError:
                        b['layers'][poly[0]] = dict(total_hits=1)
        # calculate percentage of hits for each layer
        for layer in b['mb_layers']:
            b['mb_layers'][layer]['norm_hits'] = float(b['mb_layers'][layer]['total_hits']) / args.samples_per_bin
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
    mb_totals_layer = np.zeros((num_x_bins, num_y_bins))
    for id in bins:
        b = bins[id]
        mb_totals_layer[b['id_tuple']] = b['mb_layers']['baselayer']['norm_hits'] * 0.02 + b['mb_layers']['other_layers']['norm_hits'] * 0.04
    np.save(open(args.output_name + '.npy', 'wb'), totals_layer)
    print("Showing totals layer")
    plt.imshow(totals_layer.T, origin="lower", interpolation="nearest", extent=[start.x, end.x, start.y, end.y])
    #plt.colorbar()
    # create the colorbar with a better scale to the image:
    plt.colorbar(fraction=0.015, pad=0.04)
    plt.savefig(args.output_name + '.png')
    plt.savefig(args.output_name + '.eps')
    plt.show()
    # And the material budget:
    np.save(open(args.output_name + '.mb.npy', 'wb'), mb_totals_layer)
    plt.imshow(mb_totals_layer.T, origin="lower", interpolation="nearest", extent=[start.x, end.x, start.y, end.y])
    plt.colorbar(fraction=0.015, pad=0.04)
    plt.savefig(args.output_name + '.mb.png')
    plt.savefig(args.output_name + '.mb.eps')
    plt.show()
    import pdb; pdb.set_trace()
    #for layer in per_layer_maps:
    #    print("Showing plot for layer {}".format(layer))
    #    #plt.imshow(per_layer_maps[layer], origin="lower", interpolation="gaussian",extent=[start.x, end.x, start.y, end.y])
    #    plt.imshow(per_layer_maps[layer].T, origin="lower", extent=[start.x, end.x, start.y, end.y])
    #    plt.show()

if __name__ == "__main__":
    main()

