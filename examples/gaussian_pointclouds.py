"""Example of alignment between two random noise pointclouds"""

from oarp.pcl import Pointcloud
from oarp.vis_utils import setup_axes, plot_pointcloud
import numpy as np
import os
from time import perf_counter

# make sure working from main dir
if os.getcwd().endswith('examples'):
	os.chdir('..')

np.set_printoptions(precision=3, suppress=True)

N = 1000  # Number of verts
pcl_size = 5.  # size of points in scatter

if __name__ == "__main__":

	# Gaussian pointclouds, with pcl A's major axis in x, pcl B's in y
	np.random.seed(0)
	verts_A = np.random.randn(N, 3)
	np.random.seed(1)
	verts_B = np.random.randn(N, 3)
	verts_A[..., [1, 2]] *= 0.5
	verts_B[..., [0, 2]] *= 0.5

	pcl_A = Pointcloud(verts=verts_A)
	pcl_B = Pointcloud(verts=verts_B)

	# Set up ORDER of pointcloud 1 to be by y, then by x, then z for clear visualisation
	x, y, z = pcl_A.verts.T
	order = pcl_A.order[np.lexsort((z, y, x))]
	pcl_A.order = order

	# Set up plot
	fig, axs = setup_axes(ncols=4, nrows=1, axis_opt='off', bounds=(-2., 2.))
	plot_pointcloud(axs[0], pcl_A, s=pcl_size), plot_pointcloud(axs[1], pcl_B, s=pcl_size)

	# Apply ICP and replot
	start_time = perf_counter()
	pcl_B_aligned = pcl_B.icp_align(pcl_A, max_iter=10, nsample=100, k=75)['pcl']
	print(f"Alignment... Time: {(perf_counter() - start_time) * 1000:.2f}ms")
	plot_pointcloud(axs[2], pcl_B_aligned, s=pcl_size)

	# then, perform reordering to align vertex order
	# Note: Here, neighbour selection is set quite high (100).
	# This is necessary for random pointclouds to avoid infeasible solutions.
	# Try reducing this value to see where it fails
	start_time = perf_counter()
	pcl_B_reordered = pcl_B_aligned.reorder(pcl_A, neighbours=100)['pcl']
	print(f"Reorder... Time: {(perf_counter() - start_time) * 1000:.2f}ms")

	plot_pointcloud(axs[3], pcl_B_reordered, s=pcl_size)

	titles = ['Pountcloud A', 'Pointcloud B', 'B aligned to A', 'B reordered to A']
	[ax.set_title(t) for ax, t in zip(axs, titles)]
	fig.savefig('examples/gaussian_pointclouds.png')