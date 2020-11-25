"""Functionality for reordering pointclouds"""
from oarp.pcl import Pointcloud
import numpy as np
from typing import Union
from sklearn.neighbors import NearestNeighbors

from scipy.optimize import linear_sum_assignment # for reorganising pointclouds

from itertools import chain
flatten = lambda l: list(chain.from_iterable(l))

def reorder(A: Pointcloud, B: Pointcloud, inplace=True,
			neighbours=10):
	"""Modifies the order of Pointcloud A such that the elementwise distance between the two pointclouds is
	minimised.

	inplace: By default, modifies the order of A in place
	neighbours: Number of nearest neighbours to consider for options. Set None to consider all vertices as potential
	pairings (warning: will significantly increase runtime speed)

	:returns (reordered_pcl) A Pointcloud object representing B, with a new order
	"""

	if not inplace:
		A = A.copy()

	out_order = np.zeros_like(A.order)

	# COST MATRIX, ROWS = TARGET, COL = SOURCE
	if neighbours is None:
		cost_mat = np.linalg.norm(A.verts[None, :] - B.verts[:, None], axis=-1)

	else:
		nn = NearestNeighbors(n_neighbors=neighbours, algorithm='auto').fit(B.verts)

		dst, B_idxs = nn.kneighbors(A.verts)  # get dist, indices in terms of B

		num_A = A.n_verts
		num_B = B.n_verts
		cost_mat = np.full((num_B, num_A), fill_value=np.inf) # mat of all in A -> all neighbours in B

		# Set all nearest neighbour values to their distance
		# convert idxs - vertex indices for B, to flat_inds - indices for distances in a flattened version of cost_matrix
		flat_inds = (num_A * B_idxs + np.arange(num_A)[:, None])
		np.put(cost_mat, flat_inds, dst)

	try:
		R, C = linear_sum_assignment(cost_matrix=cost_mat)
	except ValueError:
		raise ValueError("Cost matrix is infeasible. This might be due to the choice of the variable 'neighbour'.\nTry increasing it, or turn neighbour selection off by setting it to None.")

	order = C
	A.order = order

	return dict(order=C)