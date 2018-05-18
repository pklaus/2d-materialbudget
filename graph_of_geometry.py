#!/usr/bin/env python

from graphviz import Digraph

import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_geometry_file')
    parser.add_argument('--duplicates', action='store_true')
    args = parser.parse_args()

    # Starting to read in the geometry definition
    with open(args.json_geometry_file, 'r') as geometry_file:
        g = json.load(geometry_file)

    top_level_entity = g['top_level_entity']

    dot = Digraph(comment=top_level_entity)
    #dot.engine = 'circo'
    #dot.engine = 'neato'
    dot.engine = 'dot'
    nodes = []
    edges = []
    create_graph(top_level_entity, g['components'], nodes, edges)

    # draw nodes
    for node in set(nodes):
        dot.node(node, node)
    # draw edges
    if not args.duplicates:
        for edge in set(edges):
            #dot.edge(*edge, label='{}x'.format(edges.count(edge)), constraint='false')
            dot.edge(*edge, label='{}x'.format(edges.count(edge)))
    else:
        for edge in edges:
            dot.edge(*edge)

    dot.render(args.json_geometry_file + '.gv', view=True)

def create_graph(nodename, components, nodes, edges):
    node = components[nodename]
    nodes.append(node['name'])
    if 'components' not in node: return
    for subnode in node['components']:
        edge = (node['name'], subnode['name'])
        edges.append(edge)
        create_graph(subnode['name'], components, nodes, edges)

if __name__ == "__main__": main()
