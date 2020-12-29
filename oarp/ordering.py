"""Functionality for reordering pointclouds"""
from __future__ import annotations
import numpy as np
from typing import Union
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment  # for reorganising pointclouds
from itertools import chain

try:
	from sslap import auction_solve
	use_sslap = True
except ImportError:
	print("No sslap found, using scipy for ordering.")
	use_sslap = False

# import Pointcloud for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from oarp.pcl import Pointcloud

flatten = lambda l: list(chain.from_iterable(l))


def solve_lap(mat):
	"""Solve Linear Assignment Problem"""
	if use_sslap: # use sslap auction algorithm implementation
		mat[mat == np.inf] = -1
		auction_sol = auction_solve(mat, problem='min')
		sol = auction_sol['sol']
	else:
		R, sol = linear_sum_assignment(cost_matrix=mat)

	return sol


def reorder(A: Pointcloud, B: Pointcloud, inplace=True, neighbours=10):
	"""Modifies the order of Pointcloud A such that the elementwise distance between the two pointclouds is
	minimised.

	inplace: By default, modifies the order of A in place
	neighbours: Number of nearest neighbours to consider for options.
				Set to None to consider all vertices as potential
				Set to 1 to only consider closest nodes (no assignment algorithm) - useful if meshes are identical
	pairings (warning: will significantly increase runtime speed)

	:returns (reordered_pcl) A Pointcloud object representing B, with a new order
	"""

	if not inplace:
		A = A.copy()

	# COST MATRIX, ROWS = TARGET, COL = SOURCE

	if neighbours == 1:
		# simple one-to-one minimum distance assignment
		nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(A.verts)
		dst, A_idxs = nn.kneighbors(B.verts)
		C = A_idxs.ravel()

	else:
		if neighbours is None:
			cost_mat = np.linalg.norm(A.verts[None, :] - B.verts[:, None], axis=-1)

		else:
			num_A = A.n_verts
			num_B = B.n_verts
			cost_mat = np.full((num_B, num_A), fill_value=np.inf)  # mat of all in A -> all neighbours in B

			# To ensure matrix is feasible, must consider nearest neighbours for B,
			nn = NearestNeighbors(n_neighbors=neighbours, algorithm='auto').fit(B.verts)
			dst_B, B_idxs = nn.kneighbors(A.verts)  # get dist, indices in terms of B

			# but also nearest neighbours for A (to ensure that every vertex in both pointclouds maps to at least one other
			nn = NearestNeighbors(n_neighbors=neighbours, algorithm='auto').fit(A.verts)
			dst_A, A_idxs = nn.kneighbors(B.verts)  # get dist, indices in terms of B

			# Set all nearest neighbour values to their distance
			# convert idxs - vertex indices for B, to flat_inds - indices for distances in a flattened version of cost_matrix
			flat_inds = (num_A * B_idxs + np.arange(num_A)[:, None])
			np.put(cost_mat, flat_inds, dst_B)

			flat_inds = (A_idxs + num_A * np.arange(num_A)[:, None])
			np.put(cost_mat, flat_inds, dst_A)

		try:
			C = solve_lap(cost_mat)
		except ValueError:
			raise ValueError(
				"Cost matrix is infeasible. This might be due to the choice of the variable 'neighbour'.\nTry increasing it, or turn neighbour selection off by setting it to None.")

	order = C
	A.order = order

	return dict(pcl=A, order=C)
