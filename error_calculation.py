import numpy as np
import matplotlib.pyplot as plt
import sys

from math import pi, sqrt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from classifiers import classifier

class error_calc:
	@staticmethod
	def med2_error(a, b):
		med2_cm_boundary =[0 for _ in range(len(a.cluster) + len(b.cluster))]
		points = np.concatenate([a.cluster, b.cluster])
		
		for i in range(len(points)):
			a_dist = sqrt((points[i][0] - a.mean[0])**2 + (points[i][1] - a.mean[1])**2)
			b_dist = sqrt((points[i][0] - b.mean[0])**2 + (points[i][1] - b.mean[1])**2)

			if min(a_dist, b_dist) == a_dist:
				med2_cm_boundary[i] = 1
			else:
				med2_cm_boundary[i] = 2
			
			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED2 error... Row: {0:4}/{1:4}'.format(i + 1, len(med2_cm_boundary)))

		print('... completed.')
		return med2_cm_boundary
			

	@staticmethod
	def med3_error(c, d, e):
		med3_cm_boundary =[0 for _ in range(len(c.cluster) + len(d.cluster) + len(e.cluster))]
		points = np.concatenate([c.cluster, d.cluster, e.cluster])
		
		for i in range(len(points)):
			c_dist = sqrt((points[i][0] - c.mean[0])**2 + (points[i][1] - c.mean[1])**2)
			d_dist = sqrt((points[i][0] - d.mean[0])**2 + (points[i][1] - d.mean[1])**2)
			e_dist = sqrt((points[i][0] - e.mean[0])**2 + (points[i][1] - e.mean[1])**2)
			
			if min(c_dist, d_dist, e_dist) == c_dist:
				med3_cm_boundary[i] = 1
			elif min(c_dist, d_dist, e_dist) == d_dist:
				med3_cm_boundary[i] = 2
			else:
				med3_cm_boundary[i] = 3
			
			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED3 error... Row: {0:4}/{1:4}'.format(i + 1, len(med3_cm_boundary)))

		print('... completed.')
		return med3_cm_boundary


	@staticmethod
	def ged2_error(a, b):
		ged2_cm_boundary =[0 for _ in range(len(a.cluster) + len(b.cluster))]
		points_ab = np.concatenate([a.cluster, b.cluster])
		
		for i,point in enumerate(points_ab):
			a_dist = classifier.get_micd_dist(a, point)
			b_dist = classifier.get_micd_dist(b, point)

			if min(a_dist, b_dist) == a_dist:
				ged2_cm_boundary[i] = 1
			else:
				ged2_cm_boundary[i] = 2

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED3 error... Row: {0:4}/{1:4}'.format(i + 1, len(ged2_cm_boundary)))

		print('... completed.')
		return ged2_cm_boundary


	@staticmethod
	def ged3_error(c, d, e):
		ged3_cm_boundary = [0 for _ in range(len(c.cluster) + len(d.cluster) + len(e.cluster))]
		points_cde = np.concatenate([c.cluster, d.cluster, e.cluster])
		
		for i,point in enumerate(points_cde):
			c_dist = classifier.get_micd_dist(c, point)
			d_dist = classifier.get_micd_dist(d, point)
			e_dist = classifier.get_micd_dist(e, point)

			if min(c_dist, d_dist, e_dist) == c_dist:
				ged3_cm_boundary[i] = 1
			elif min(c_dist, d_dist, e_dist) == d_dist:
				ged3_cm_boundary[i] = 2
			else:
				ged3_cm_boundary[i] = 3
			
			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED3 error... Row: {0:4}/{1:4}'.format(i + 1, len(ged3_cm_boundary)))

		print('... completed.')
		return ged3_cm_boundary


	@staticmethod
	def nn2_test(a, b):
		nn2_cm_boundary = [0 for _ in range(len(a.testing_cluster) + len(b.testing_cluster))]
		points_ab = np.concatenate([a.testing_cluster, b.testing_cluster])
		
		for i,point in enumerate(points_ab):
			a_dist = float('inf')
			for coord in a.cluster:
				temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], point[0], point[1])
				if temp_dist < a_dist:
					a_dist = temp_dist

			b_dist = float('inf')
			for coord in b.cluster:
				temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], point[0], point[1])
				if temp_dist < b_dist:
					b_dist = temp_dist

			if min(a_dist, b_dist) == a_dist:
				nn2_cm_boundary[i] = 1
			else:
				nn2_cm_boundary[i] = 2

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating NN2 error...  Row: {0:4}/{1:4}'.format(i + 1, len(nn2_cm_boundary)))

		print('... completed.')
		return nn2_cm_boundary


	@staticmethod
	def nn3_test(c, d, e):
		nn3_cm_boundary = [0 for _ in range(len(c.testing_cluster) + len(d.testing_cluster) + len(e.testing_cluster))]
		points_cde = np.concatenate([c.testing_cluster, d.testing_cluster, e.testing_cluster])
		
		for i,point in enumerate(points_cde):
			# Find nearest neighbours
			c_dist = float('inf')
			for coord in c.cluster:
				temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], point[0], point[1])
				if temp_dist < c_dist:
					c_dist = temp_dist

			d_dist = float('inf')
			for coord in d.cluster:
				temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], point[0], point[1])
				if temp_dist < d_dist:
					d_dist = temp_dist

			e_dist = float('inf')
			for coord in e.cluster:
				temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], point[0], point[1])
				if temp_dist < e_dist:
					e_dist = temp_dist

			if min(c_dist, d_dist, e_dist) == c_dist:
				nn3_cm_boundary[i] = 1
			elif min(c_dist, d_dist, e_dist) == d_dist:
				nn3_cm_boundary[i] = 2
			else:
				nn3_cm_boundary[i] = 3

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating NN3 error...  Row: {0:4}/{1:4}'.format(i + 1, len(nn3_cm_boundary)))

		print('... completed.')
		return nn3_cm_boundary


	@staticmethod
	def knn2_test(a, b):
		knn2_cm_boundary = [0 for _ in range(len(a.testing_cluster) + len(b.testing_cluster))]
		points_ab = np.concatenate([a.testing_cluster, b.testing_cluster])
		
		for i, point in enumerate(points_ab):
				# Find nearest neighbours
				a_group = [float('inf') for _ in range(4)]
				for coord in a.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], point[0], point[1])
					if temp_dist < max(a_group):
						a_group[a_group.index(max(a_group))] = temp_dist

				a_dist = np.mean(a_group)

				b_group = [float('inf') for _ in range(4)]
				for coord in b.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], point[0], point[1])
					if temp_dist < max(b_group):
						b_group[b_group.index(max(b_group))] = temp_dist

				b_dist = np.mean(b_group)

				if min(a_dist, b_dist) == a_dist:
					knn2_cm_boundary[i] = 1
				else:
					knn2_cm_boundary[i] = 2

				# Print progress
				sys.stdout.write('\r')
				sys.stdout.write('Calculating KNN2 error... Row: {0:4}/{1:4}'.format(i + 1, len(knn2_cm_boundary)))

		print('... completed.')
		return knn2_cm_boundary


	@staticmethod
	def knn3_test(c, d, e):
		knn3_cm_boundary = [0 for _ in range(len(c.testing_cluster) + len(d.testing_cluster) + len(e.testing_cluster))]
		points_cde = np.concatenate([c.testing_cluster, d.testing_cluster, e.testing_cluster])
		
		for i, point in enumerate(points_cde):
			# Find nearest neighbours
			c_group = [float('inf') for _ in range(4)]
			for coord in c.cluster:
				temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], point[0], point[1])
				if temp_dist < max(c_group):
					c_group[c_group.index(max(c_group))] = temp_dist

			c_dist = np.mean(c_group)

			d_group = [float('inf') for _ in range(4)]
			for coord in d.cluster:
				temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], point[0], point[1])
				if temp_dist < max(d_group):
					d_group[d_group.index(max(d_group))] = temp_dist

			d_dist = np.mean(d_group)

			e_group = [float('inf') for _ in range(4)]
			for coord in e.cluster:
				temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], point[0], point[1])
				if temp_dist < max(e_group):
					e_group[e_group.index(max(e_group))] = temp_dist

			e_dist = np.mean(e_group)

			if min(c_dist, d_dist, e_dist) == c_dist:
				knn3_cm_boundary[i] = 1
			elif min(c_dist, d_dist, e_dist) == d_dist:
				knn3_cm_boundary[i] = 2
			else:
				knn3_cm_boundary[i] = 3

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating KNN3 error... Row: {0:4}/{1:4}'.format(i + 1, len(knn3_cm_boundary)))

		print('... completed.')
		return knn3_cm_boundary