"""Example of alignment between two cacti obj files.

Two stage: ICP, then reordering"""

from oarp.pcl import Pointcloud
from oarp.vis_utils import setup_axes, plot_pointcloud
import numpy as np
from time import perf_counter
import os

# make sure working from main dir
if os.getcwd().endswith('examples'):
	os.chdir('..')

np.set_printoptions(precision=3, suppress=True)
obj_src = 'meshes/stanford_bunny_lp.obj'


if __name__ == "__main__":
	pcl_orig = Pointcloud.load_from_obj(obj_src)
	print(f"LOADED MESH: {obj_src} [{pcl_orig.n_verts} verts]")

	# Set up ORDER of pointcloud 1 to be x, then z, then y for clear visualisation
	x, y, z = pcl_orig.verts.T
	order = pcl_orig.order[np.lexsort((y, z, x))]
	pcl_orig.order = order

	# Copy pointcloud
	pcl_transformed = pcl_orig.copy()

	# apply desired transforms - rotate, shift, and reorder
	pcl_transformed.rotate_about_axis(0.5, 'z')
	pcl_transformed.rotate_about_axis(0.5, 'y')
	pcl_transformed.rotate_about_axis(0.5, 'x')
	pcl_transformed.shift_in(-0.6, 'y')
	pcl_transformed.randomise()

	# Set up plot of two pointclouds
	bounds = (pcl_orig.bbox[:, 0].min(), pcl_orig.bbox[:, 1].max())
	fig, axs = setup_axes(ncols=4, axis_opt='off', bounds=bounds, elev=25, azim=60)
	plot_pointcloud(axs[0], pcl_orig)
	plot_pointcloud(axs[1], pcl_transformed)

	## ALIGNMENT OF PCL_TRANSFORMED -> PCL_ALIGN
	# First, try pure PCA
	start_time = perf_counter()
	pca_fit = pcl_transformed.pca_align(pcl_orig)
	_, pca_fit = pca_fit['pcl'], pca_fit['meta']  # get data from ICP results
	print(f"PCA...  Time: {(perf_counter() - start_time) * 1000:.2f}ms | Error : {np.format_float_scientific(pca_fit['dst'], 3)}")

	# Next, perform pure ICP (no initial guess)
	start_time = perf_counter()
	icp_fit = pcl_transformed.icp_align(pcl_orig, max_iter=100, nsample=100, k=95, T_init=np.eye(4))
	pcl_realigned, icp_meta = icp_fit['pcl'], icp_fit['meta'] # get data from ICP results
	print('ICP... ', f"Time: {(perf_counter() - start_time) * 1000:.2f}ms | Num its: {icp_meta['nits']} | Error : {np.format_float_scientific(icp_meta['dst'], 3)}")

	# PCA runs faster, but ICP converges slightly more accurately. Take ICP result as the aligned pointcloud
	plot_pointcloud(axs[2], pcl_realigned)

	# Then, perform reordering to align vertex order.
	# Here, we know the meshes are identical, so we could set neighbours = 1 for super fast fitting
	# Using neighbours = 10 to show the speed of the linear assignment algorithm
	start_time = perf_counter()
	reorder_res = pcl_realigned.reorder(pcl_orig, neighbours=5)
	pcl_reordered, reorder_meta = reorder_res['pcl'], reorder_res['meta']
	print(f"Reorder... Time: {(perf_counter() - start_time) * 1000:.2f}ms")


	# Prove that all vertsicesand vertex order are identical
	print(f"Vertices match: {np.allclose(pcl_orig.verts, pcl_reordered.verts)}")
	print(f"Vertex order matches: {np.allclose(pcl_orig.order, pcl_reordered.order)}")

	plot_pointcloud(axs[3], pcl_reordered)

	titles = ['Original', 'Transformed & shuffled', 'Realigned', 'Reordered']
	[ax.set_title(t) for ax, t in zip(axs, titles)]
	fig.savefig('examples/rigid_transforms.png')
