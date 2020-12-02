OARP
----
Library for **O**rdering and **A**lignment of **R**igid **P**ointclouds.

![](examples/rigid_transforms.png)

This library provides tools for rigid alignment and vertex reordering in order to configure two pointclouds. These pointclouds can be, as above, from the same source mesh, with transformations applied (allowing for a perfect fit), or any two pointclouds, demonstrated below, in which case a best-fit alignment and reordering will be found.

![](examples/gaussian_pointclouds.png)

# Examples

To run the either of the examples demonstrated in the images above, run either:

- Transformed pointcloud ordering and alignment: `python examples/rigid_pointclouds.py`
- Dissimilar pointcloud ordering and alignment: `python examples/gaussian_pointclouds.py`

# Implementation

All implementations are in numpy and scipy.

Alignment is completed via Principal Component Analysis (PCA) and Iterative Closest Point (ICP). 

PCA provides an initial estimate for the rigid transformation between two pointclouds (highly accurate if these are from the same source), and ICP iteratively refines this estimate. 