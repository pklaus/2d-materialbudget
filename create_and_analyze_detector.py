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
import time
from datetime import datetime as dt
from itertools import combinations
import logging

if shapely.speedups.available:
    shapely.speedups.enable()

clock = time.perf_counter
logger = logging.getLogger('create-and-analyze-detector')

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
        #if self.relevant_polygons: logger.info('there are %d relevant polygons!', len(self.relevant_polygons))
        for poly in self.relevant_polygons:
            if poly.contains(other): return True
        return False

    def overlap(self, other):
        area = 0.0
        intersecting_polygons = []
        pid = multiprocessing.current_process().pid
        start = clock()
        logger.debug("#{pid} {time:.3f} starting to calc intersections on {num_relevant} relevant polygons".format(pid=pid, time=(clock()-start), num_relevant=len(self.relevant_polygons)))
        for poly in self.relevant_polygons:
            intersection = other.intersection(poly)
            if not intersection.is_empty:
                intersecting_polygons.append(poly)
                area += intersection.area
        logger.debug("#{pid} {time:.3f} intersections calculated. Correcting {num_intersecting} intersecting polygons now.".format(pid=pid, time=(clock()-start), num_intersecting=len(intersecting_polygons)))
        for combo in combinations(intersecting_polygons, 2):
            intersection = combo[0].intersection(combo[1])
            if not intersection.is_empty:
                area -= intersection.area
        logger.debug("#{pid} {time:.3f} intersections corrected".format(pid=pid, time=(clock()-start)))
        return area

    def set_relevant_polygons(self, shape):
        len_total = len(self.polygons)
        self.relevant_polygons = [p for p in self.polygons if p.intersects(shape)]
        len_relevant = len(self.relevant_polygons)
        logger.debug("{} component: selected {} out of {} polygons as relevant".format(self.name, len_relevant, len_total))
        return len_relevant > 0

def apply_transforms(base_components, transforms):
    for transform in transforms:
        logger.debug('Performing a transform of kind "{kind}"'.format(**transform))
        t = dict(xfact=1.0, yfact=1.0, xoff=0.0, yoff=0.0, origin='center')
        t.update(dict(transform))
        t = Munch(t)
        for base_component in base_components:
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
        logger.debug('Performing a transform of kind "{kind}"'.format(**transform))
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
    def __init__(self, patch, geometry, relevant_base_components, strategy='sample'):
        self.patch = patch
        self.geometry = geometry
        self.relevant_base_components = relevant_base_components
        self.strategy = strategy

    def calc(self):
        start = clock()
        pid = multiprocessing.current_process().pid
        logger.debug("{id} (patch id) {pid} (process id) starting calculating patch".format(pid=pid, **self.patch))
        logger.debug("calculating for position {}".format(self.patch.shape.centroid))
        r = dict() # result dict
        p = self.patch
        shape = p.shape
        bounds = shape.bounds
        #bcs = self.geometry.components[self.geometry.top_level_entity].base_components
        bcs = self.relevant_base_components
        logger.debug("#{pid} {time:.3f} - relevant basic shapes: {num_bcs}".format(time=(clock()-start), pid=pid, num_bcs=len(bcs)))
        if self.strategy == 'sample':
            for sample in p.samples:
                # calculate sample positions
                sp = shapely.geometry.Point(sample[0] * p.width_step + bounds[0], sample[1] * p.height_step + bounds[1])
                for bc in bcs:
                    #logger.info('bc: %s point: %s', bc, sp)
                    #logger.info('number of relevant polygons: %d', len(bc.relevant_polygons))
                    if bc.contains(sp):
                        try:
                            r[bc.material] += bc.material_budget
                        except KeyError:
                            r[bc.material] = bc.material_budget
            # normalize material budget to number of samples
            for material in r:
                r[material] = float(r[material]) / len(p.samples)
        elif self.strategy == 'calculate':
            shape_area = shape.area
            for bc in bcs:
                overlap = bc.overlap(shape)
                if overlap > 0.0:
                    try:
                        r[bc.material] += bc.material_budget * (overlap / shape_area)
                    except KeyError:
                        r[bc.material] = bc.material_budget * (overlap / shape_area)
        else:
            raise NotImplementedError('strategy: ' + str(self.strategy))
        logger.debug("#{pid} {time:.3f} overlap calculated".format(time=(clock()-start), pid=pid))

        self.result = Munch(r)
        del bcs
        del self.geometry
        logger.debug("{id} (patch id) {pid} (process id) finished calculating patch after {time:.3f}s".format(time=clock()-start, pid=pid, **self.patch))
        return self

def parse_point(string):
    coords = string.split(',')
    return shapely.geometry.Point(float(coords[0]), float(coords[1]))

def is_power_of_two(num): return ((num & (num - 1)) == 0) and num > 0

def main():
    import argparse

    def argparse_power_of_two(value):
        num = int(value)
        if not is_power_of_two(num):
            raise argparse.ArgumentTypeError(f"'{value}' is not a power of two.")
        return num

    parser = argparse.ArgumentParser(description="Create a detector SVG image from a geometry file stating its components")
    parser.add_argument('--debug', action='store_true', help='enable debugging output')
    parser.add_argument('--start', metavar='POINT', required=True, type=parse_point, help='Start Point')
    parser.add_argument('--end',   metavar='POINT', required=True, type=parse_point, help='End Point')
    parser.add_argument('--num-x-bins', type=argparse_power_of_two, required=True, help='Number of bins in the x direction (should be a power of 2)')
    parser.add_argument('--num-y-bins', type=argparse_power_of_two, required=True, help='Number of bins in the y direction (should be a power of 2)')
    parser.add_argument('--samples-per-bin', type=int, help='Number of bins in the y direction')
    parser.add_argument('--top-level-entity', help='Defines the top level entity (overrides the one stated in the geometry file)')
    parser.add_argument('--acceptance-radius', type=float, default=None, help='Acceptance radius [mm]. (Defaults to tan(25 deg) * --distance-to-target)')
    parser.add_argument('--distance-to-target', type=float, default=None, help='Distance between the detector (station) and target [mm].')
    parser.add_argument('--strategy', choices=('sample', 'calculate'), help='Stategy to determine the materialbudget.')
    parser.add_argument('json_geometry_file', help='The geometry file in JSON format')
    parser.add_argument('output_name', help='The basename for the output files')
    args = parser.parse_args()

    # setting up logging and considering --debug
    ch = logging.StreamHandler()
    fh = logging.FileHandler(args.output_name + '.log', 'w+')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)

    args_dict = vars(args)
    start = clock()
    run(**args_dict)
    duration = clock() - start
    logger.info("Total run time: %.3fs", duration)


def run(**kwargs):
    # Save the kwargs in a file:
    args_dict = kwargs.copy()
    args_dict['start'] = str(args_dict['start'])
    args_dict['end'] = str(args_dict['end'])
    with open(kwargs.get('output_name') + '.args.pydict', 'w') as f:
        f.write(pprint.pformat(args_dict))
    shutil.copyfile(kwargs.get('json_geometry_file'), kwargs.get('output_name') + '.geometry.json')

    # Let's make sure that our bounds follow those rules:
    #   --start.x  <  --end.x   and   --start.y  <  --end.y
    f, t = kwargs.get('start'), kwargs.get('end')
    start = shapely.geometry.Point(min(f.x, t.x), min(f.y, t.y))
    end   = shapely.geometry.Point(max(f.x, t.x), max(f.y, t.y))
    del f, t
    logger.info('Imaging area from {} to {}'.format(start, end))

    num_x_bins = kwargs.get('num_x_bins')
    num_y_bins = kwargs.get('num_y_bins')
    assert is_power_of_two(num_x_bins)
    assert is_power_of_two(num_y_bins)
    width  = end.x - start.x
    height = end.y - start.y
    width_step  = width  / num_x_bins
    height_step = height / num_y_bins

    # Starting to read in the geometry definition
    with open(kwargs.get('json_geometry_file'), 'r') as geometry_file:
        g = json.load(geometry_file)
    # g stands for 'geometry'
    g = munchify(g)

    if kwargs.get('top_level_entity', None):
        g.top_level_entity = kwargs.get('top_level_entity')

    if kwargs.get('distance_to_target', None):
        g.distance_to_target = kwargs.get('distance_to_target')

    if kwargs.get('acceptance_radius', None):
        g.acceptance_radius = kwargs.get('acceptance_radius')
    else:
        g.acceptance_radius = 0.46630766 * g.distance_to_target # 0.46630766 == tan( 25 deg )

    cs = g.components

    # Set .base_components for the components of kind == polygon
    poly_components = [c for c in cs if cs[c].kind == 'polygon']
    for poly_component in poly_components:
        logger.info("Reading polygon {}".format(poly_component))
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
        logger.debug("SVG file: %s", svg_file)
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

    logger.info("Calculating component tree")
    calc_base_components(g, g.top_level_entity)
    logger.info("Converting to SVG coordinate space")
    apply_transforms(g.components[g.top_level_entity].base_components, [dict(kind='scale', yfact=-1, origin=(0, 0))])
    logger.info("Creating SVG")
    dwg = geometry_to_svg(g, g.top_level_entity, g.size, [int(s/2) for s in g.size], profile='tiny')
    logger.info("Saving to SVG")
    dwg.saveas(kwargs.get('output_name') + '.svg')
    logger.info("Finished saving the SVG file!")


    # Preparing our Patches (= bins)
    logger.info("Preparing the patches (bins)")
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
        p.samples = np.random.rand(kwargs.get('samples_per_bin'), 2)
        p.materials = dict()
        p.components = dict()
        patches[id] = p

    #def div_conq(x_idx, y_idx, x_pos, y_pos):
    #    global TOTAL_STARTED_FUNCS
    #    TOTAL_STARTED_FUNCS += 1
    #    print(x_idx, y_idx, "starting; total started funcs:", TOTAL_STARTED_FUNCS)
    #    def split_list(l):
    #        "split list with even number of entries in two halfs"
    #        half_len = len(l)//2
    #        return l[:half_len], l[half_len:]
    #    if len(x_idx) == 1 and len(y_idx) == 1:
    #        # done dividing, let's conquer
    #        print("x_pos", x_pos[x_idx[0]], "y_pos", y_pos[y_idx[0]])
    #        TOTAL_STARTED_FUNCS -= 1
    #        print(x_idx, y_idx, "ending; new amount of total started funcs:", TOTAL_STARTED_FUNCS)
    #        return
    #    if len(x_idx) >= len(y_idx):
    #        x_idx_left, x_idx_right = split_list(x_idx)
    #        div_conq(x_idx_left, y_idx, x_pos, y_pos)
    #        div_conq(x_idx_right, y_idx, x_pos, y_pos)
    #    else:
    #        y_idx_left, y_idx_right = split_list(y_idx)
    #        div_conq(x_idx, y_idx_left, x_pos, y_pos)
    #        div_conq(x_idx, y_idx_right, x_pos, y_pos)
    #    TOTAL_STARTED_FUNCS -= 1
    #    print(x_idx, y_idx, "ending; new amount of total started funcs:", TOTAL_STARTED_FUNCS)


    # Calculating the "jobs"
    logger.info("Creating the CalculatePatchJob()s by calculating the relevant base components in divide and conquer style")
    jobs = []
    for id in patches:
        bcs = g.components[g.top_level_entity].base_components
        relevant_bcs = []
        for bc in bcs:
            relevant = bc.set_relevant_polygons(patches[id].shape)
            if relevant:
                bc = copy.copy(bc)
                relevant_bcs.append(bc)
                #logger.info('there are %d relevant polygons in bc %s', len(bc.relevant_polygons), bc)
        #if relevant_bcs: logger.info('there are %d relevant bcs!', len(relevant_bcs))
        bcs = relevant_bcs
        jobs.append(CalculatePatchJob(patches[id], g, relevant_bcs, strategy=kwargs.get('strategy')))

    # Starting 'imaging'
    logger.info("Setting up the multiprocessing process pool")
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    chunksize = 3 * multiprocessing.cpu_count()
    #import sys, statistics
    #sizes = [sys.getsizeof(job) for job in jobs]
    #stats = {
    # "mean": statistics.mean(sizes),
    # "min":  min(sizes),
    # "max":  max(sizes),
    # "median": statistics.median(sizes),
    # }
    #logger.debug("Job statistics/sizeof: mean: {mean} median: {median} min: {min} max: {max}".format(**info))
    #sys.exit()
    #jobs = p.map(calc_job, jobs, chunksize=chunksize)
    ## instead use imap_unordered (to give more status output):
    results = []
    start_time = dt.now()
    logger.info("Starting the job processing (material budget sampling in each bin)")
    for i, result in enumerate(p.imap_unordered(calc_job, jobs, chunksize)):
        now = dt.now()
        total_time = (now-start_time)/(i+1) * len(jobs)
        eta = start_time + total_time # estimated time of arrival
        eta = eta.replace(microsecond=0)
        logger.info("Done with CalculatePatchJob #{} of {} ({:.1%}). ETA: {} ETT: {}".format(i, len(jobs), i/len(jobs), eta, total_time))
        results.append(result)
    jobs = results
    p.close()

    logger.info("Saving the results")
    # save the CalculatePatchJob results:
    for job in jobs:
        del job.patch.samples
    with open(kwargs.get('output_name') + '.pkl', 'wb') as f:
        pickle.dump(jobs, f, pickle.HIGHEST_PROTOCOL)

    # Reporting
    totals_layer = np.zeros((num_x_bins, num_y_bins))
    for job in jobs:
        p = job.patch
        for material in job.result:
            totals_layer[p.id_tuple] += job.result[material]
    np.save(open(kwargs.get('output_name') + '.npy', 'wb'), totals_layer)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.67)
    plt.imshow(totals_layer, origin="lower", norm=norm, interpolation="nearest", extent=[start.x, end.x, start.y, end.y])
    plt.gca().add_patch(plt.Circle((0, 0), radius=g.acceptance_radius, fill=None, edgecolor='r'))
    # create the colorbar with a better scale to the image:
    plt.colorbar(fraction=0.015, pad=0.04)
    plt.savefig(kwargs.get('output_name') + '.png')
    plt.savefig(kwargs.get('output_name') + '.eps')
    plt.show()


if __name__ == "__main__":
    main()

