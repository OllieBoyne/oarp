"""Example of alignment between two cacti obj files.

Two stage: ICP, then reordering"""

from oarp.pcl import Pointcloud
from oarp.icp import ICP
from oarp.reordering import reorder
from oarp.vis_utils import setup_axes, plot_pointcloud, draw_arrow
import numpy as np
from time import perf_counter
from matplotlib import pyplot as plt

np.set_printoptions(precision=3, suppress=True)

obj_src = 'meshes/bunny_LP.obj'
# obj_src_1 = 'meshes/cactus.obj'

if __name__ == "__main__":
	pcl_1 = Pointcloud.load_from_obj(obj_src)
	print(f"LOADED MESH: {obj_src} [{pcl_1.n_verts} verts]")

	# Set up ORDER of pointcloud 1 to be by y, then by x, then z for clear visualisation
	x, y, z = pcl_1.verts.T

	order = pcl_1.order[np.lexsort((z, y, x))]
	pcl_1.order = order

	# Generate second pointcloud, with transformations applied, and indices reordered
	pcl_2 = pcl_1.copy()

	# apply transform
	T = np.array([	[0, -1, 0, 1.2],
					[1, 0, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1]	])

	pcl_2.transform(T)
	pcl_2.randomise()  # add in to test reordering

	d = 0.5
	fig, axs = setup_axes(ncols=2, nrows=2, axis_opt='off', bounds=pcl_1.bbox)

	plot_pointcloud(axs[0], pcl_1)
	plot_pointcloud(axs[1], pcl_2)

	# First, perform ICP to align pcl_2 with pcl_1

	start_time = perf_counter()
	res = ICP(pcl_2, pcl_1, max_iter=10, nsample=100, k=75)
	print('ICP... ', f"Time: {(perf_counter() - start_time) * 1000:.2f}ms | Num its: {res['nits']} | Error : {np.format_float_scientific(res['err'], 3)}")

	plot_pointcloud(axs[2], pcl_2)

	# then, perform reordering to align vertex order
	start_time = perf_counter()
	reres = reorder(pcl_2, pcl_1, neighbours=10)
	print(f"Reorder... Time: {(perf_counter() - start_time) * 1000:.2f}ms")

	# Prove that all verts and vertex order are identical
	print(f"Vertices match: {np.allclose(pcl_1.verts, pcl_2.verts)}")
	print(f"Vertex order matches: {np.allclose(pcl_1.order, pcl_2.order)}")

	plot_pointcloud(axs[3], pcl_2)

	draw_arrow(axs[0], axs[1], (1., 0.5), (0., 0.5), text='Translate, rotate, shuffle')
	draw_arrow(axs[1], axs[2], (0, 0.), (1., 1.), text='Realign')
	draw_arrow(axs[2], axs[3], (1., 0.5), (0., 0.5), text='Reorder')

	plt.show()
