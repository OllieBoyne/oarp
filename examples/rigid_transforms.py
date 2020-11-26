"""Example of alignment between two cacti obj files.

Two stage: ICP, then reordering"""

from oarp.pcl import Pointcloud
from oarp.icp import ICP
from oarp.reordering import reorder
from oarp.vis_utils import setup_axes, plot_pointcloud
import numpy as np
from time import perf_counter
import os

from matplotlib import pyplot as plt

# make sure working from main dir
if os.getcwd().endswith('examples'):
	os.chdir('..')

np.set_printoptions(precision=3, suppress=True)
obj_src = 'meshes/stanford_bunny_lp.obj'

if __name__ == "__main__":
	pcl_1 = Pointcloud.load_from_obj(obj_src)
	print(f"LOADED MESH: {obj_src} [{pcl_1.n_verts} verts]")

	# Set up ORDER of pointcloud 1 to be x, then z, then y for clear visualisation
	x, y, z = pcl_1.verts.T
	order = pcl_1.order[np.lexsort((y, z, x))]
	pcl_1.order = order

	# Copy pointcloud
	pcl_2 = pcl_1.copy()

	# apply desired transforms - rotate, shift, and reorder
	pcl_2.rotate_about_axis(0.5, 'z')
	pcl_2.rotate_about_axis(0.5, 'y')
	pcl_2.shift_in(-0.6, 'y')
	pcl_2.randomise()

	fig, axs = setup_axes(ncols=4, axis_opt='off', bounds=pcl_1.bbox, elev=25, azim=60)

	plot_pointcloud(axs[0], pcl_1)
	plot_pointcloud(axs[1], pcl_2)

	# First, perform ICP to align pcl_2 with pcl_1

	start_time = perf_counter()
	res = ICP(pcl_2, pcl_1, max_iter=20, nsample=100, k=75)
	print('ICP... ', f"Time: {(perf_counter() - start_time) * 1000:.2f}ms | Num its: {res['nits']} | Error : {np.format_float_scientific(res['err'], 3)}")

	plot_pointcloud(axs[2], pcl_2)

	# then, perform reordering to align vertex order
	start_time = perf_counter()
	reres = reorder(pcl_2, pcl_1, neighbours=20)
	print(f"Reorder... Time: {(perf_counter() - start_time) * 1000:.2f}ms")

	# Prove that all verts and vertex order are identical
	print(f"Vertices match: {np.allclose(pcl_1.verts, pcl_2.verts)}")
	print(f"Vertex order matches: {np.allclose(pcl_1.order, pcl_2.order)}")

	plot_pointcloud(axs[3], pcl_2)

	titles = ['Original', 'Transformed & shuffled', 'Realigned', 'Reordered']
	[ax.set_title(t) for ax, t in zip(axs, titles)]
	fig.savefig('examples/rigid_transforms.png')
