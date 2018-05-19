#!/usr/bin/env python

# A script to create material budget plots

import json

class HistoPlot(object):
    def __init__(self, n, f, t):
        """
        n : number of bins
        f : from
        t : to
        """
        raise NotImplementedError()

    def fill(self, values):
        raise NotImplementedError()

    def draw(self):
        raise NotImplementedError()

class MatplotlibPlot(HistoPlot):
    def __init__(self, n, f, t):
        import numpy as np
        from matplotlib import pyplot as plt
        self.plt = plt
        self.np = np

    def fill(self, values):
        #Stack the data
        #plt.figure()
        fig, ax = self.plt.subplots()
        x, y = self.np.arange(len(values)), values
        ax.stackplot(x, y)

    def draw(self):
        self.plt.show()

class RootPlot(HistoPlot):
    def __init__(self, n, f, t):
        from rootpy.plotting import Hist, Canvas
        self.canvas = Canvas()
        self.canvas.SetGrid()

        self.hist = Hist(n, f, t)
        self.hist.fillstyle = 'solid'
        self.hist.fillcolor = '#87cefa'
        #self.hist.SetFillColor(2)
        self.hist.GetXaxis().SetTitle("position on carrier [um]")
        self.hist.GetYaxis().SetTitle("Material Budget X/X_{0} %")
        #self.hist.SetTitle("Material Budget for MVD Stations 0/1")
        self.hist.SetTitle("Material Budget for MVD Stations 2/3")
        self.hist.SetStats(False) # deactivate the stats Pave

        #self.canvas.Update()
        #self.canvas.Print('./plot.pdf')

    def fill(self, values):
        i = 0
        for value in values:
            self.hist[i] = value
            i += 1

    def draw(self):
        self.hist.Draw()
        #self.hist.DrawCopy()
        #self.hist.SetFillColor(2)
        #self.hist.GetXaxis().SetRange(2, 90)
        #self.hist.DrawCopy()

class Component(object):
    def __init__(self, component_dict, material_dict):
        self.name     = component_dict['name']
        self.start_at = component_dict['start_at']
        try:
            self.end_at = component_dict['end_at']
        except KeyError:
            self.end_at = self.start_at + component_dict['width']
        self.thickness = component_dict['thickness']
        self.material = material_dict
        self.budget = self.thickness * self.material['budget_per_um']

    def is_at(self, position):
        return position >= self.start_at and position < self.end_at

    def budget_at(self, position):
        if self.is_at(position):
            return self.budget
        else:
            return 0.0
def main():
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gfile', metavar='GEOMETRY_FILE', help='The geometry to analyze (has to be JSON).')
    parser.add_argument('--from', '-f', type=float, default=0.0, help='Starting position.')
    parser.add_argument('--to', '-t', type=float, default=150000.0, help='Ending position.')
    parser.add_argument('--num-bins', '-n', type=int, default=200, help='The number of bins to analyze from starting to ending position.')
    parser.add_argument('--debug', '-d', action='store_true', help='Print more status output')

    args = parser.parse_args()
    dargs = vars(args)

    with open(args.gfile, 'r') as gfile:
        geometry = json.load(gfile)

    materials = dict()
    for dmat in geometry['materials']:
        if args.debug: pprint(dmat)
        materials[dmat['name']] = dmat

    components = []
    for dcomp in geometry['components']:
        if args.debug: pprint(dcomp)
        components.append(Component(dcomp, materials[dcomp['material']]))

    full_range = dargs['to'] - dargs['from']
    stepsize = full_range / (args.num_bins + 1)

    values = []
    position = dargs['from'] + stepsize/2
    while position <= args.to:
        budget = 0.0
        for comp in components:
            budget += comp.budget_at(position)
        if args.debug: print("Budget at the current position {:0.0f} is: {:0.2f}".format(position, budget))
        values.append(budget)
        position += stepsize

    non_zero_values = [value for value in values if value > 0.0]
    average_budget = sum(non_zero_values)/len(non_zero_values)
    print("The average material budget is {:0.2f}".format(average_budget))

    h = MatplotlibPlot(args.num_bins, dargs['from'], dargs['to'])
    h.fill(values)
    h.draw()

    raw_input()

if __name__ == "__main__":
    main()

