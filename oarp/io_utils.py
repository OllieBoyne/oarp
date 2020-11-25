"""Utils for loading and saving different filetypes"""
import os

class Loader:

	def load(self, src):
		with open(src, 'r') as infile:
			return self._read(infile)

	def save(self, src, data):
		with open(src, 'w') as outfile:
			self._write(outfile, data)

	def _read(self, infile):
		raise NotImplementedError

	def _write(self, outfile, data):
		raise NotImplementedError

class OBJLoader(Loader):

	def _read(self, infile):
		verts = []
		faces = []
		for line in infile.readlines():
			if line[:2] == 'v ':
				verts.append(list(map(float, line.split(" ")[1:])))

			if line[:2] == 'v ':
				# f  v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
				face = []
				for vset in line.split(" ")[1:]:
					face.append(vset.split("/")[0])

				faces.append(face)

		return verts, faces
