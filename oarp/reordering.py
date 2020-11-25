"""Functionality for reordering pointclouds"""
from oarp.pcl import Pointcloud
import numpy as np
from typing import Union
from sklearn.neighbors import NearestNeighbors

from scipy.optimize import linear_sum_assignment # for reorganising pointclouds

from itertools import chain
flatten = lambda l: list(chain.from_iterable(l))

def run_partition(A: Pointcloud, B: Pointcloud, nn:NearestNeighbors, cost_matrix:np.ndarray, splits=(5, 0, 0)):
	"""
	Partitions A into fixed regions, before running linear_sum_assignment on each partitition, to speed up
	assignments

	:param nn: scikit NearestNeighbour function which returns idxs of nearest neighbours in B to all in A
	:param B:  The target pcl
	:param cost_matrix: Distances shape (B.n_verts, A.n_verts). inf for 'unconnected' points
	:param splits: int or (x,y,z,) tuple, with number of splits per dimension
	:return: r, c: row, column assignment for linear sum assignment
	"""

	partitions = A.partition_verts(splits)


def reorder(A: Pointcloud, B: Pointcloud, inplace=True, partition=True,
			neighbours=10):
	"""Modifies the order of Pointcloud A such that the elementwise distance between the two pointclouds is
	minimised.

	inplace: By default, modifies the order of A in place
	partition: Split cost matrix into sections based on regions of pointcloud. Speed savings
	neighbours: Number of nearest neighbours to consider for options. Set None to consider all vertices as potential
	pairings (warning: will significantly increase runtime speed)

	:returns (reordered_pcl) A Pointcloud object representing B, with a new order
	"""

	if not inplace:
		A = A.copy()

	# COST MATRIX, ROWS = TARGET, COL = SOURCE
	if neighbours is None:
		cost_mat = np.linalg.norm(A.verts[None, :] - B.verts[:, None], axis=-1)

	else:
		nn = NearestNeighbors(n_neighbors=neighbours, algorithm='auto').fit(B.verts)
		dst, idxs = nn.kneighbors(A.verts)
		cost_mat = np.full((A.n_verts, B.n_verts), fill_value=np.inf)

		# Set all nearest neighbour values to their distance
		# convert idxs - vertex indices for B, to flat_inds - indices for distances in a flattened version of cost_matrix
		flat_inds = (A.n_verts * idxs + np.arange(A.n_verts)[:, None])
		np.put(cost_mat, flat_inds, dst)

	# run_partition(A, B, nn, cost_mat)

	R, C = linear_sum_assignment(cost_matrix=cost_mat)
	order = C
	A.order = order

	return dict(order=C)