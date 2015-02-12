
### 2d-materialbudget

A collection of tools to analyze material budget in two dimensions

#### Coordinate System

This software uses the following coordinate system definition:

* *x* points 'rightwards'
* *y* points 'upwards'

The units used in this package denote 'mm'
(if that's of any importance for your application).

#### SVG support

These tools support reading SVG files.
There are a couple of things to keep in mind though:

* SVG only has a notion of pixels (px).
  One SVG px translates into one mm in this software.
* In the SVG coordinate system, however, y points downwards.
  So whenever we import from or export to SVG,
  we have to transform the coordinates accordingly.

This software has some limitations concerning SVG files.

* The supported shapes are polygons without bezier curves and rectangles.
* The IDs of the shapes have to follow a certain naming convention.
* Do not use transforms of your shapes.
  This can happen quickly in Inkscape, e.g. when you group several
  shapes and then resize the group, or when you rotate an object.

#### Author

* Philipp Klaus
  <klaus@physik.uni-frankfurt.de>

