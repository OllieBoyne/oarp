"""Pointcloud class, with additional utilities"""
import numpy as np
from oarp.io_utils import OBJLoader
from sklearn.decomposition import PCA
import copy
from typing import Union


def apply_transform(verts, T):
	"""Apply 4x4 transformation matrix to verts"""
	v = np.pad(verts, ((0, 0), (0, 1)), constant_values=1)
	return (v @ T.T)[..., :3]


def in_bbox(V, bbox):
	"""Returns a boolean array of shape V = (..., 3), corresponding to whether each point in V
	is within the (3, 2) bbox """
	x, y, z = V[..., 0], V[..., 1], V[..., 2]
	return (bbox[0, 0] <= x) & (x < bbox[0, 1]) & (bbox[1, 0] <= y) & (y < bbox[1, 1]) & (bbox[2, 0] <= z) & (
				z < bbox[2, 1])


class Pointcloud():
	def __init__(self, verts: np.ndarray):
		"""Verts: [N x 3] pointclouds"""

		self._verts = verts
		self.n_verts = len(verts)

		self._order_modified = False  # flag for if order is changed
		self._order = np.arange(self.n_verts)  # default order

		self.pose = np.eye(4)

	@property
	def order(self):
		return self._order

	@order.setter
	def order(self, order):
		self._order_modified = True
		self._order = self._order[order]

	@property
	def verts(self):
		templ_verts = self._verts[self._order]
		return apply_transform(templ_verts, self.pose)

	@property
	def verts_padded(self):
		return np.pad(self.verts, ((0, 0), (0, 1)), constant_values=1)

	def transform(self, T: np.ndarray):
		"""Apply rigid transformation T to current pose.
		:param T: 4x4 np.ndarray, depciting [R|t], where R is a translation matrix, and t a translation"""

		if not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5):
			raise ValueError("The top left 3x3 matrix of T must represent purely a rotation matrix.")

		self.pose = T @ self.pose

	def rotate_about_axis(self, angle: float, axis='x'):
		"""Apply rigid rotation to current pose.
		:param angle: Rotation in radians
		:param axis: Letter corresponding to axis of rotation, x, y or z"""
		T = np.eye(4)
		rot_2d = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
		idxs = [[1, 2], [0, 2], [0, 1]]['xyz'.index(axis)]
		T[np.ix_(idxs, idxs)] = rot_2d
		self.transform(T)

	def shift_in(self, shift: float, axis='x'):
		"""Apply shift to current pose
		:param shift: Float to shift
		:param axis: Axis to shift in, x, y, z
		"""
		T = np.eye(4)
		T['xyz'.index(axis), -1] = shift
		self.transform(T)

	@staticmethod
	def load_from_obj(src):
		loader = OBJLoader()

		verts, faces = loader.load(src)
		pcl = Pointcloud(verts=np.array(verts))
		return pcl

	def randomise(self):
		"""Randomise pointcloud order (used mainly for illustration purposes)"""
		self._order_modified = True
		np.random.shuffle(self._order)

	def copy(self):
		return copy.deepcopy(self)

	@property
	def centroid(self):
		return self.verts.mean(axis=0)

	@property
	def bbox(self):
		"""Return (3 x 2) bounding box of pointcloud"""
		return np.stack([self.verts.min(axis=0), self.verts.max(axis=0)], axis=-1)

	def get_principal_axes(self):
		"""Return (3,3) principal axes, in order of moment"""
		norm_verts = self.verts - self.verts.mean(axis=0)
		pca = PCA(n_components=3).fit(norm_verts).components_

		# to ensure cyclic set, force vec 3 to be the cross product of the first 2
		pca[2] = np.cross(pca[0], pca[1])

		return pca

	def partition_verts(self, splits: Union[int, tuple] = 2):
		"""Return a list of verts lists, each made by splitting the pointcloud up in 3 dimensions.
		if splits is int, splits that many times in each dimension
		if splits is tuple, split by splits[0] in x, splits[1] in y, etc

		returns:
			partition: list of verts lists
			idxs : list of idxs lists
		"""

		if isinstance(splits, int):
			splits = (splits, splits, splits)

		# calculate partition lines in each dimension
		bbox = self.bbox
		xp = np.linspace(bbox[0, 0], bbox[0, 1] + 1e-5, splits[0] + 2)
		yp = np.linspace(bbox[1, 0], bbox[1, 1] + 1e-5, splits[1] + 2)
		zp = np.linspace(bbox[2, 0], bbox[2, 1] + 1e-5, splits[2] + 2)

		partition, idxs = [], []
		for xi in range(splits[0] + 1):
			for yi in range(splits[1] + 1):
				for zi in range(splits[2] + 1):
					p_bbox = np.array([xp[xi:xi + 2], yp[yi:yi + 2], zp[zi:zi + 2]])
					in_bounds = in_bbox(self.verts, p_bbox)
					partition.append(self.verts[in_bounds])
					idxs.append(np.argwhere(in_bounds).ravel())

		return partition, idxs
