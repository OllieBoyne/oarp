"""Example of alignment between two random noise pointclouds"""

from oarp.pcl import Pointcloud
from oarp.icp import ICP
from oarp.reordering import reorder
import numpy as np

from matplotlib import pyplot as plt
from oarp.vis_utils import setup_axes, plot_pointcloud, draw_arrow

np.set_printoptions(precision=3, suppress=True)

N = 1000

if __name__ == "__main__":
	pcl_1 = Pointcloud(verts=np.random.randn(N, 3))
	pcl_2 = Pointcloud(verts=np.random.randn(N, 3))

	# Set up ORDER of pointcloud 1 to be by y, then by x, then z for clear visualisation
	x, y, z = pcl_1.verts.T
	order = pcl_1.order[np.lexsort((z, y, x))]
	pcl_1.order = order

	fig, axs = setup_axes(ncols=2, nrows=2, axis_opt='off', bounds=pcl_1.bbox)

	plot_pointcloud(axs[0], pcl_1),	plot_pointcloud(axs[1], pcl_2)

	# Apply ICP and replot
	res = ICP(pcl_2, pcl_1, max_iter=10, nsample=100, k=75)
	plot_pointcloud(axs[2], pcl_2)

	# then, perform reordering to align vertex order
	# Note: Here, neighbour selection is turned off. This comes at a cost to performance,
	# but might be necessary for random pointclouds, as limiting to nearest neighbours might remove feasible solutions
	reres = reorder(pcl_2, pcl_1, neighbours=None)

	plot_pointcloud(axs[3], pcl_2)

	plt.show()