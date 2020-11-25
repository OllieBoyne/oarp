"""Implementation of iterative closest point (ICP) algorithm in numpy"""
from oarp.pcl import Pointcloud
import numpy as np
from typing import Union
from sklearn.neighbors import NearestNeighbors
import cv2

#TODO: check how reflecting transforms are handled

from time import perf_counter
class Timer:
	def __init__(self):
		self.log = {}
		self.t0 = perf_counter()

	def time(self, label):
		elapsed = perf_counter() - self.t0
		self.log[label] = self.log.get(label, []) + [elapsed]
		self.t0 = perf_counter()

	def get_log(self):
		print("LOG: ")
		[print(f"{k}: {1000*np.mean(v):.2f}") for k,v in self.log.items()]


# def _ax_sizes(pcl: Pointcloud):
# 	"""Return (3,) size vector of size of axis in each dimension"""
# 	bbox = pcl.bbox
# 	return bbox[:, 1] - bbox[:, 0]
#
# def _rank_axes(pcl: Pointcloud):
# 	"""Return (3,) size vector of order of x,y,z axes, from largest to smallest in pointcloud size"""
# 	sizes = _ax_sizes(pcl)
# 	ranks = np.arange(3)[sizes.argsort()][::-1]
# 	return ranks
#
# def _bulk_align_pointclouds(src_pointcloud: Pointcloud, target_pointcloud: Pointcloud):
# 	"""Returns the 4x4 affine transformation matrix that provides a bulk alignment of xyz 'axes' in order of size,
# 	and translation, to provide an initial guess for ICP"""


def estimate_bulk_transform(src_pointcloud: Pointcloud, target_pointcloud: Pointcloud):
	"""Using principal axes and centroids, identify the affine matrix T (4x4) that translates the axes
	of src_pointcloud to match that of target_pointcloud"""

	S, Q = src_pointcloud.get_principal_axes(), target_pointcloud.get_principal_axes()
	R = Q.T @ S

	T = np.eye(4)
	T[:3, :3] = R
	T[:3, -1] = (target_pointcloud.centroid - R @ src_pointcloud.centroid.T)

	return T

#http://resources.mpi-inf.mpg.de/deformableShapeMatching/EG2012_Tutorial/slides/1.2%20ICP_+_TPS_%28NM%29.pdf

# or http://ais.informatik.uni-freiburg.de/teaching/ss12/robotics/slides/17-icp.pdf

def ICP(src_pointcloud: Pointcloud, target_pointcloud: Pointcloud, T_init=None,
		max_iter=30, tol=1e-8, nsample=1000, k=75, inplace=True):
	"""Identifies the affine transformation matrix T, that best aligns the unordered pointcloud
	src_pointcloud, with the target_pointcloud.

	src_pointcloud and target_pointcloud are Pointcloud objects
	if inplace is True src_pointcloud will be transformed in place
	:param nsample: Number of veritces randomly selected for comparison each iteration
	:param T_init: Optional initial guess for T. (4x4) np.ndarray
	:param k: Only evaluate the k% nearest of the nearest neighbour pairs

	Returns dict:
		T = 4x4 transformation matrix
		nits = number of iterations occured
		err = Average nearest neighbour error after final iteration

	Runs until max_iter or if reduction in error between iterations drops below tol, whichever comes first
	"""

	if not inplace:
		src_pointcloud = src_pointcloud.copy()

	target_verts = target_pointcloud.verts
	nn = NearestNeighbors(n_neighbors=1, algorithm='auto',
						  ).fit(target_verts)

	pose_init = src_pointcloud.pose # get initial pose for recalculation

	if T_init is None:
		# make initial guess, based on PCA and centroids
		T = estimate_bulk_transform(src_pointcloud, target_pointcloud)
	else:
		T = np.array(T_init)

	src_pointcloud.transform(T)

	prev_err = None
	n = 0

	for n in range(max_iter):

		# VARIANTS TO IMPLEMENT
		# Point subsets
		# weighted correspondences
		# data association
		# rejecting outliers
		# - test with variable number of verts in src/targ

		# Select sample from source mesh
		sample = np.random.randint(0, src_pointcloud.n_verts, nsample)
		src = src_pointcloud.verts[sample]

		# identify corresponding NN on target mesh
		dst, idxs = nn.kneighbors(src) # get idxs of nearest neighbours on source
		targ = target_verts[idxs.ravel()]

		# reject pairs with distance > k times median
		filt = dst.ravel() <= np.percentile(dst.ravel(), k) #k * np.median(dst.ravel())
		src, targ = src[filt], targ[filt]

		err = np.abs(dst).mean()

		if n > 0 and abs(err - prev_err) <= tol:
			break

		src_centroid, targ_centroid = src.mean(axis=0), targ.mean(axis=0)
		src_delta = src - src_centroid
		targ_delta = targ - targ_centroid
		W = np.dot(src_delta.T, targ_delta)

		U, S, VT = np.linalg.svd(W)
		R = (U @ VT).T

		# for reflections
		if np.linalg.det(R) < 0:
			VT[2, :] *= -1
			R = np.dot(VT.T, U.T)

		t = targ_centroid - (R @ src_centroid.T).T
		T = np.eye(4)
		T[:3, :3] = R
		T[:3, -1] = t

		# apply transformation
		src_pointcloud.transform(T)
		prev_err = err

	dst, idxs = nn.kneighbors(src_pointcloud.verts)  # final distances
	res = dict(T=src_pointcloud.pose @ np.linalg.inv(pose_init), nits=n+1, err=dst.mean())

	return res