"""Example of alignment between two cacti obj files.

Two stage: ICP, then reordering"""

from oarp.pcl import Pointcloud
from oarp.icp import ICP
from oarp.reordering import reorder
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

np.set_printoptions(precision=3, suppress=True)

# obj_src_1 = 'meshes/bunny.obj'
obj_src_1 = 'meshes/cactus.obj'


def draw_arrow(ax1, ax2, xy1=(0,0), xy2=(0,0), text=None):
	"""Draw arrow between axes, with xy1, xy2 in axes frac coords"""
	con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA='axes fraction' , coordsB='axes fraction' ,
						  axesA=ax1, axesB=ax2, color='blue', arrowstyle="-|>")

	if text is not None:
		xy = np.mean([ax1.transAxes.transform(xy1), ax2.transAxes.transform(xy2)],
					 axis=0)
		plt.annotate(text, xy, xycoords='figure pixels', va='center')

	ax1.add_artist(con)


if __name__ == "__main__":
	pcl_1 = Pointcloud.load_from_obj(obj_src_1)

	# Set up ORDER of pointcloud 1 to be by y, then by x, then z for clear visualisation
	x, y, z = pcl_1.verts.T
	
	order = pcl_1.order[np.lexsort((z, y, x))]
	pcl_1.order = order

	# Generate second pointcloud, with transformations applied, and indices reordered
	pcl_2 = pcl_1.copy()

	# apply transform
	T = np.array( [ [0, -1, 0, 0.5],
				  	[1, 0, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1]])

	pcl_2.transform(T)
	pcl_2.randomise() # add in to test reordering

	# pcl_2._verts += np.random.randn(*
	# 		pcl_2.verts.shape) / 100

	fig, axs = plt.subplots(ncols=2, nrows=2, subplot_kw=dict(projection='3d'))
	axs = axs.ravel() # [orig, distorted, realigned, reordered]
	[ax.axis('off') for ax in axs]

	d = 0.5
	for ax in axs:
		ax.set_xlim(-d, d)
		ax.set_ylim(-d, d)
		ax.set_zlim(-d, d)

	axs[0].scatter(*pcl_1.verts.T, alpha=1., c=np.arange(pcl_1.n_verts))
	axs[1].scatter(*pcl_2.verts.T, alpha=1., c=np.arange(pcl_2.n_verts))

	# First, perform ICP to align pcl_2 with pcl_1
	from time import perf_counter
	start_time = perf_counter()
	res = ICP(pcl_2, pcl_1, max_iter=10, nsample=100, k=75
			  )
	print('ICP', f"{(perf_counter()-start_time)*1000:.2f}ms")

	axs[2].scatter(*pcl_2.verts.T, c = np.arange(pcl_2.n_verts)) # Show ICP result

	# print("EXPECTED T: ")
	# print(np.linalg.inv(T))
	# print("ICP T: ")
	# print(res['T'])
	# print("NIT", res['nits'])
	# print("Err", res['err'])

	# then, perform reordering to align vertex order
	start_time = perf_counter()
	reres = reorder(pcl_2, pcl_1, neighbours=5)
	print('reorder', f"{(perf_counter()-start_time)*1000:.2f}ms")

	# assignment working, but not order
	print(np.allclose(pcl_1.verts, pcl_2.verts))
	print(np.allclose(pcl_1.order, pcl_2.order))

	axs[3].scatter(*pcl_2.verts.T, c = np.arange(pcl_2.n_verts)) # Show reorder result

	draw_arrow(axs[0], axs[1], (1., 0.5), (0., 0.5), text='Translate, rotate, shuffle')
	draw_arrow(axs[1], axs[2], (0, 0.), (1., 1.), text='Realign')
	draw_arrow(axs[2], axs[3], (1., 0.5), (0., 0.5), text='Reorder')


	plt.show()