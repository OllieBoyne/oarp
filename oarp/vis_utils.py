import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from typing import Union

from matplotlib.patches import ConnectionPatch

title_font = {'family': 'serif', 'serif': ['Arial'], 'weight': 'bold', 'size': 22}
mpl.rc('font', **title_font)
default_cmap = 'cividis'


def setup_axes(ncols=1, nrows=1, axis_opt='off', bounds: Union[float, int, tuple, np.ndarray] = 1,
			   azim=60, elev=30):
	"""
	:param ncols: Number of columns
	:param nrows: Number of rows
	:param axis_opt: Axis 3D options eg 'on', 'off' (equal & others may not work for 3D axes) *
	:param bounds: Bounds for axes, either as:
			- int: all axes are plotted from -bounds -> bounds
			- tuple/ndarray, size (2,): all axes are plotted from bounds[0] -> bounds[1]
			- ndarray, size(3,2): axis i is plotted from bounds[i, 0] -> bounds[i, 1]
	:return: flattened axes, np.ndarray (ncols*nrows,)

	*equal, scaled, tight, auto, image, square
	"""

	fig, axs = plt.subplots(ncols=ncols, nrows=nrows, subplot_kw=dict(projection='3d'))
	axs = axs.ravel()
	[ax.axis(axis_opt) for ax in axs]  # set option

	# identify bounds
	_bounds = np.zeros((3, 2))

	if isinstance(bounds, int) or isinstance(bounds, float):
		_bounds[:, 0], _bounds[:, 1] = -bounds, bounds

	elif isinstance(bounds, tuple) or (isinstance(bounds, np.ndarray) and bounds.size == 2):
		_bounds[0] = bounds
		_bounds[1] = _bounds[2] = _bounds[0]

	elif isinstance(bounds, np.ndarray) and bounds.shape == (3, 2):
		_bounds = bounds

	else:
		raise NotImplementedError(f"Bounds of type {type(bounds)} not implemented here.")

	# apply bounds
	for ax in axs:
		ax.set_xlim(*_bounds[0])
		ax.set_ylim(*_bounds[2])  # axis 'y' takes data z
		ax.set_zlim(*_bounds[1])  # axis 'z' takes data y
		ax.view_init(azim=azim, elev=elev)

	# select aspect ratio as ~ rows/columns (slightly increased to account for titles)
	aspect = mpl.figure.figaspect(1.2 * nrows / ncols)

	fig.set_size_inches(aspect)
	fig.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)

	return fig, axs

def plot_pointcloud(ax, pcl, alpha=1., c=None, cmap=default_cmap):
	"""Note - 'z' axis on graph maps the y coordinate here, and 'y' maps the z coordinate, to be consistent
	with the .obj file format"""
	if c is None:
		c = np.arange(pcl.n_verts)  # by default, colour based on order

	ax.scatter(*pcl.verts[:, [0, 2, 1]].T, c=c, alpha=alpha, cmap=cmap)  # plot x, z, y


def draw_arrow(ax1, ax2, xy1=(0, 0), xy2=(0, 0), text=None):
	"""Draw arrow between axes, with xy1, xy2 in axes frac coords"""
	con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA='axes fraction', coordsB='axes fraction',
						  axesA=ax1, axesB=ax2, color='blue', arrowstyle="-|>")

	if text is not None:
		xy = np.mean([ax1.transAxes.transform(xy1), ax2.transAxes.transform(xy2)], axis=0)
		plt.annotate(text, xy, xycoords='figure pixels', va='center')

	ax1.add_artist(con)
