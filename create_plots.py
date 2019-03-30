#!/usr/bin/env python

# Local dependencies
from create_and_analyze_detector import BaseComponent, CalculatePatchJob

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

def main():
    global DEBUG

    import argparse
    parser = argparse.ArgumentParser(description="Circular-calc")
    parser.add_argument('--title', help='An (optional) title for the plots')
    parser.add_argument('--distance-to-target', type=float, required=True, help='Distance between the detector (station) and target [mm].')
    parser.add_argument('--debug', action='store_true', help='enable debugging output')
    parser.add_argument('input_file', help='The input file (Pickle / .pkl file)')
    parser.add_argument('output_name', help='The basename for the output files')
    args = parser.parse_args()

    DEBUG = args.debug

    dist_target = args.distance_to_target
    title = args.title

    with open(args.input_file, 'rb') as f:
        jobs = pickle.load(f)

    ###### Theta Plot ######

    def bin_idx(pos, bin_limits):
        #if pos < bin_start or pos > bin_end:
        if pos < bin_limits[0] or pos > bin_limits[-1]:
            return None
        i = 0
        while pos > bin_limits[i+1]:
            i += 1
        return i


    num_bins = 62
    bin_start = 0
    bin_end = 30
    angles = np.linspace(bin_start, bin_end, num=(num_bins+1))
    values = dict()
    hits = np.zeros(num_bins)
    for job in jobs:
        pos = job.patch.shape.centroid
        rdist = (pos.x**2+pos.y**2)**(.5)
        theta = math.atan(rdist/dist_target)*180/math.pi
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
    order = ['diamond', 'CVD', 'TPG', 'glue', 'Al', 'Si', 'polyimide', 'Cu']
    width = angles[1:] - angles[:-1]
    angles = angles[:-1]
    materials = []
    for material in order:
        if material not in values: continue
        materials.append(material)
    for material in values.keys():
        if material not in materials: materials.append(material)
    for material in materials:
        values[material] = values[material]/hits
        plots[material] = plt.bar(angles, values[material], width=width, bottom=bottom, color=colors[color_index])
        bottom += values[material]
        color_index += 1
        color_index %= 9
    axis_font = {'size': '16'}
    #axis_font = {'fontname':'Arial', 'size':'14'}
    plt.ylabel(r'Material budget $\mathrm{x/X_0}\ [\%]$',     **axis_font)
    plt.xlabel(r'Polar angle $\mathrm{\vartheta}\ [^\circ]$', **axis_font)
    if title: plt.title(title)
    title_font = {'size': '22'}
    #plt.title('3rd MVD station with PRESTO', **title_font)
    #plt.text(r'integrated over the azimuthal angle $\varphi$')
    lrefs = [plots[material][0] for material in materials]
    plt.legend(reversed(lrefs), reversed(materials), loc=2)
    # Limit 0.3
    #material_budget_limit = 0.3
    # Limit 0.5
    material_budget_limit = 0.5
    #col_mb_limit = 'g'
    col_mb_limit = (.0, .7, .0)
    #col_accept   = 'y'
    col_accept   = (.93, .93, .0)
    plt.gca().add_line(plt.Line2D((2.5, 25), (material_budget_limit, material_budget_limit), lw=2, color=col_mb_limit )) # color='r'))
    plt.gca().add_line(plt.Line2D((25, 25), (0, material_budget_limit), lw=2, color=col_accept))
    plt.gca().add_line(plt.Line2D((2.5, 2.5), (0, material_budget_limit), lw=2, color=col_accept))

    x = (0, 2.5,                   2.5,                    25, 25, 30,  30,   0)
    y = (0, 0,   material_budget_limit, material_budget_limit,  0,  0, 0.6, 0.6)
    plt.fill(x, y, alpha=0.4, facecolor='w')
    #patch(Polygon([[0, 0], [4, 1.1], [6, 2.5], [2, 1.4]], closed=True, fill=False, hatch='/'))

    plt.ylim(top=0.63, bottom=0.0)

    plt.savefig(args.output_name + '.theta.png')
    plt.savefig(args.output_name + '.theta.eps')
    plt.show()
    plt.close()


    #####  x-y-plot (sum over all materials)  #####
    num_x_bins, num_y_bins = 0, 0
    for job in jobs:
        num_x_bins = max(num_x_bins, job.patch.x_num)
        num_y_bins = max(num_y_bins, job.patch.y_num)
    num_x_bins += 1; num_y_bins += 1
    totals_layer = np.zeros((num_x_bins, num_y_bins))
    extents_x = []
    extents_y = []
    for job in jobs:
        p = job.patch
        extents_x.append(job.patch.shape.exterior.coords.xy[0])
        extents_y.append(job.patch.shape.exterior.coords.xy[1])
        for material in job.result:
            totals_layer[p.id_tuple] += job.result[material]
    extent_x = min(min(vals) for vals in extents_x), max(max(vals) for vals in extents_x)
    extent_y = min(min(vals) for vals in extents_y), max(max(vals) for vals in extents_y)
    extent = (extent_x[1], extent_x[0], extent_y[0], extent_y[1]) # (also inverts x axis)
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
    #norm = matplotlib.colors.Normalize()
    #if material_budget_limit == 0.3:
    #    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.402)
    #elif material_budget_limit == 0.5:
    #    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.67)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.67)
    #plt.imshow(totals_layer, origin="lower", interpolation="nearest", extent=[start.x, end.x, start.y, end.y])
    #plt.gca().add_patch(plt.Circle((0, 0), radius=g.acceptance_radius, fill=None, edgecolor='r'))
    plt.imshow(totals_layer, cmap=cmap, norm=norm, aspect='equal', origin="lower", interpolation="nearest", extent=extent)
    radius = dist_target * 0.46630766 # 0.46630766 == tan(25 deg)
    plt.gca().add_patch(plt.Circle((0, 0), radius=radius, fill=None, edgecolor='r'))
    # dist_target ==  5: extent=[-30, 30, -30, 30])
    # dist_target == 10: extent=[-55, 55, -55, 55])
    # dist_target == 15: extent=[-80, 80, -80, 80])
    # dist_target == 20: extent=[-105, 105, -105, 105])

    # create the colorbar with a better scale to the image:
    plt.colorbar(fraction=0.015, pad=0.04)
    plt.ylabel(r'Y coordinate [mm]', **axis_font)
    plt.xlabel(r'X coordinate [mm]', **axis_font)
    plt.title(title)
    plt.savefig(args.output_name + '.png')
    plt.savefig(args.output_name + '.eps')
    plt.show()
    plt.close()


    #####  y-slice plot  #####

    values = dict()
    hits = np.zeros(num_bins)
    y_values = []
    for job in jobs:
        pos = job.patch.shape.centroid
        y_values.append(pos.x)
    most_central_x = min(y_values, key=lambda x: abs(x - 0.0))
    x_values = []
    materials = []
    for job in jobs:
        pos = job.patch.shape.centroid
        if pos.x == most_central_x:
            x_values.append(pos.y)
            materials += list(job.result.keys())
    materials = set(materials)
    mat_budget = {material: np.zeros(len(x_values)) for material in materials}
    idx = 0
    for job in jobs:
        pos = job.patch.shape.centroid
        if pos.x == most_central_x:
            for material in job.result:
                if mat_budget[material][idx]: print('Bad. Non-zero value detected: %s' % mat_budget[material][idx])
                mat_budget[material][idx] = job.result[material]
            idx += 1
    bottom = np.zeros(len(x_values))
    color_index = 0
    colors = ['#727272', '#f1595f', '#885555', '#f9a65a', '#b87333', '#599ad3', '#79c36a', '#9e66ab', '#cd7058', '#d77fb3', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plots = dict()
    order = ['diamond', 'CVD', 'TPG', 'glue', 'Al', 'Si', 'polyimide', 'Cu']
    width = angles[1:] - angles[:-1]
    angles = angles[:-1]
    materials = []
    for material in order:
        if material not in mat_budget: continue
        materials.append(material)
    for material in mat_budget.keys():
        if material not in materials: materials.append(material)
    x_values_sorted = sorted(x_values)
    width = x_values_sorted[1] - x_values_sorted[0]
    for material in materials:
        mat_budget[material] = mat_budget[material]
        plots[material] = plt.bar(x_values, mat_budget[material], width=width, bottom=bottom, color=colors[color_index])
        bottom += mat_budget[material]
        color_index += 1
        color_index %= 9
    axis_font = {'size': '16'}
    #axis_font = {'fontname':'Arial', 'size':'14'}
    plt.ylabel(r'Material budget $\mathrm{x/X_0}\ [\%]$',     **axis_font)
    plt.xlabel(r'X coordinate (y=0) [mm]', **axis_font)
    plt.xlim(left=max(x_values), right=min(x_values))
    plt.ylim(top=0.5, bottom=0.0)
    title_font = {'size': '22'}
    if title: plt.title(title, **title_font)
    lrefs = [plots[material][0] for material in materials]
    plt.legend(reversed(lrefs), reversed(materials), loc=2)

    plt.savefig(args.output_name + '.y-slice.png')
    plt.savefig(args.output_name + '.y-slice.eps')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

