import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from math import pi, sqrt, exp
from statistics import mode
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from classifiers import classifier

class error_calc:

	@staticmethod
	def get_marg(cl, point):
			point_mean_diff = (np.subtract(point, cl.mean))
			mult = np.matmul(np.transpose(point_mean_diff), (np.matmul(np.linalg.inv(cl.covariance), point_mean_diff)))
			return (1 / (((2 * pi) ** (cl.n / 2)) * sqrt(np.linalg.det(cl.covariance)))) * exp((-1 / 2) * mult)

	@staticmethod
	def med2_error(a, b, points_ab):
		start_time = time.time()
		boundary = [0 for _ in range(len(a.cluster) + len(b.cluster))]
		points = np.concatenate([a.cluster, b.cluster])

		for i in range(len(points)):
			a_dist = sqrt((points[i][0] - a.mean[0]) ** 2 + (points[i][1] - a.mean[1]) ** 2)
			b_dist = sqrt((points[i][0] - b.mean[0]) ** 2 + (points[i][1] - b.mean[1]) ** 2)

			if min(a_dist, b_dist) == a_dist:
				boundary[i] = 1
			else:
				boundary[i] = 2

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED2 error... Row: {0:4}/{1:4}'.format(i + 1, len(boundary)))

		# Confusion Matrix of MED2
		c_matrix = confusion_matrix(points_ab, boundary)

		# Error Rate of MED2
		error_rate = 1 - (accuracy_score(points_ab, boundary, normalize=True))  # error rate = 1 - accuracy score

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return c_matrix, error_rate

	@staticmethod
	def med3_error(c, d, e, points_cde):
		start_time = time.time()
		boundary = [0 for _ in range(len(c.cluster) + len(d.cluster) + len(e.cluster))]
		points = np.concatenate([c.cluster, d.cluster, e.cluster])

		for i in range(len(points)):
			c_dist = sqrt((points[i][0] - c.mean[0]) ** 2 + (points[i][1] - c.mean[1]) ** 2)
			d_dist = sqrt((points[i][0] - d.mean[0]) ** 2 + (points[i][1] - d.mean[1]) ** 2)
			e_dist = sqrt((points[i][0] - e.mean[0]) ** 2 + (points[i][1] - e.mean[1]) ** 2)

			if min(c_dist, d_dist, e_dist) == c_dist:
				boundary[i] = 1
			elif min(c_dist, d_dist, e_dist) == d_dist:
				boundary[i] = 2
			else:
				boundary[i] = 3

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED3 error... Row: {0:4}/{1:4}'.format(i + 1, len(boundary)))

		# Confusion Matrix for MED3
		c_matrix = confusion_matrix(points_cde, boundary)

		# Calculate Error Rate for MED3
		error_rate = 1 - (accuracy_score(points_cde, boundary, normalize=True))  # error rate = 1 - accuracy score

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return c_matrix, error_rate

	@staticmethod
	def ged2_error(a, b, points_ab):
		start_time = time.time()
		boundary = [0 for _ in range(len(a.cluster) + len(b.cluster))]
		points = np.concatenate([a.cluster, b.cluster])

		for i, point in enumerate(points):
			a_dist = classifier.get_micd_dist(a, point)
			b_dist = classifier.get_micd_dist(b, point)

			if min(a_dist, b_dist) == a_dist:
				boundary[i] = 1
			else:
				boundary[i] = 2

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED3 error... Row: {0:4}/{1:4}'.format(i + 1, len(boundary)))

		# Confusion Matrix for GED2
		c_matrix = confusion_matrix(points_ab, boundary)

		# Calculate Error Rate for GED2
		error_rate = 1 - (accuracy_score(points_ab, boundary, normalize=True))  # error rate = 1 - accuracy score

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return c_matrix, error_rate

	@staticmethod
	def ged3_error(c, d, e, points_cde):
		start_time = time.time()
		boundary = [0 for _ in range(len(c.cluster) + len(d.cluster) + len(e.cluster))]
		points = np.concatenate([c.cluster, d.cluster, e.cluster])

		for i, point in enumerate(points):
			c_dist = classifier.get_micd_dist(c, point)
			d_dist = classifier.get_micd_dist(d, point)
			e_dist = classifier.get_micd_dist(e, point)

			if min(c_dist, d_dist, e_dist) == c_dist:
				boundary[i] = 1
			elif min(c_dist, d_dist, e_dist) == d_dist:
				boundary[i] = 2
			else:
				boundary[i] = 3

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED3 error... Row: {0:4}/{1:4}'.format(i + 1, len(boundary)))

		# Confusion Matrix for GED3
		c_matrix = confusion_matrix(points_cde, boundary)

		# Calculate Error Rate for GED3
		error_rate = 1 - (accuracy_score(points_cde, boundary, normalize=True))  # error rate = 1 - accuracy score

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return c_matrix, error_rate

	@staticmethod
	def map2_error(a, b, points_ab):
		boundary = [0 for _ in range(len(a.cluster) + len(b.cluster))]
		points = np.concatenate([a.cluster, b.cluster])

		# Calculate P(a) and P(b)
		p_a = a.n / (a.n + b.n)
		p_b = b.n / (a.n + b.n)

		threshold = p_b / p_a

		# Get the marginal given a class
		def get_marg(cl, point):
			point_mean_diff = (np.subtract(point, cl.mean))
			mult = np.matmul(np.transpose(point_mean_diff), (np.matmul(np.linalg.inv(cl.covariance), point_mean_diff)))
			return (1 / (((2 * pi) ** (cl.n / 2)) * sqrt(np.linalg.det(cl.covariance)))) * exp((-1 / 2) * mult)

		for i, point in enumerate(points):
				a_marg = get_marg(a, point)
				b_marg = get_marg(b, point)

				boundary[i] = 1 if (a_marg / b_marg) > (p_b / p_a) else 2

				# Print progress
				sys.stdout.write('\r')
				sys.stdout.write('Calculating MAP2 error... Row: {0:4}/{1:4}'.format(i + 1, len(boundary)))

		# Confusion Matrix for MAP2
		c_matrix = confusion_matrix(points_ab, boundary)

		# Calculate Error Rate for MAP2
		error_rate = 1 - (accuracy_score(points_ab, boundary, normalize=True))  # error rate = 1 - accuracy score

		print('... completed.')
		return c_matrix, error_rate

	@staticmethod
	def map3_error(c, d, e, points_cde):
		start_time = time.time()
		boundary = [0 for _ in range(len(c.cluster) + len(d.cluster) + len(e.cluster))]
		points = np.concatenate([c.cluster, d.cluster, e.cluster])

		p_c = c.n / (c.n + c.n)
		p_d = d.n / (d.n + d.n)
		p_e = e.n / (e.n + e.n)

		for i, point in enumerate(points):
			c_marg = error_calc.get_marg(c, point)
			d_marg = error_calc.get_marg(d, point)
			e_marg = error_calc.get_marg(e, point)

			res = [1 if (c_marg / d_marg) > (p_d / p_c) else 2,
				   1 if (c_marg / e_marg) > (p_e / p_c) else 3,
				   2 if (d_marg / e_marg) > (p_e / p_d) else 3]

			boundary[i] = mode(res)

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MAP3 error... Row: {0:4}/{1:4}'.format(i + 1, len(boundary)))
		
		# Confusion Matrix for MAP3
		c_matrix = confusion_matrix(points_cde, boundary)

		# Calculate Error Rate for MAP3
		error_rate = 1 - (accuracy_score(points_cde, boundary, normalize=True))  # error rate = 1 - accuracy score

		print('... completed.')
		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return c_matrix, error_rate

	@staticmethod
	def nn2_test_error(a, b, testing_points_ab):
		start_time = time.time()
		boundary = [0 for _ in range(len(a.testing_cluster) + len(b.testing_cluster))]
		points = np.concatenate([a.testing_cluster, b.testing_cluster])

		for i, point in enumerate(points):
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
				boundary[i] = 1
			else:
				boundary[i] = 2

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating NN2 error...  Row: {0:4}/{1:4}'.format(i + 1, len(boundary)))

		# Confusion Matrix for NN2
		c_matrix = confusion_matrix(testing_points_ab, boundary)

		# Calculate Error Rate for NN2
		error_rate = 1 - (accuracy_score(testing_points_ab, boundary, normalize=True))  # error rate = 1 - accuracy score

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return c_matrix, error_rate

	@staticmethod
	def nn3_test_error(c, d, e, testing_points_cde):
		start_time = time.time()
		boundary = [0 for _ in range(len(c.testing_cluster) + len(d.testing_cluster) + len(e.testing_cluster))]
		points = np.concatenate([c.testing_cluster, d.testing_cluster, e.testing_cluster])

		for i, point in enumerate(points):
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
				boundary[i] = 1
			elif min(c_dist, d_dist, e_dist) == d_dist:
				boundary[i] = 2
			else:
				boundary[i] = 3

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating NN3 error...  Row: {0:4}/{1:4}'.format(i + 1, len(boundary)))

		# Confusion Matrix for NN3
		c_matrix = confusion_matrix(testing_points_cde, boundary)

		# Calculate Error Rate for NN3
		error_rate = 1 - (accuracy_score(testing_points_cde, boundary, normalize=True))  # error rate = 1 - accuracy score

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return c_matrix, error_rate

	@staticmethod
	def knn2_test_error(a, b, testing_points_ab):
		start_time = time.time()
		boundary = [0 for _ in range(len(a.testing_cluster) + len(b.testing_cluster))]
		points = np.concatenate([a.testing_cluster, b.testing_cluster])

		for i, point in enumerate(points):
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
				boundary[i] = 1
			else:
				boundary[i] = 2

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating KNN2 error... Row: {0:4}/{1:4}'.format(i + 1, len(boundary)))

		# Confusion Matrix for KNN2
		c_matrix = confusion_matrix(testing_points_ab, boundary)

		# Calculate Error Rate for KNN2
		error_rate = 1 - (accuracy_score(testing_points_ab, boundary, normalize=True))  # error rate = 1 - accuracy score

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return c_matrix, error_rate

	@staticmethod
	def knn3_test_error(c, d, e, testing_points_cde):
		start_time = time.time()
		boundary = [0 for _ in range(len(c.testing_cluster) + len(d.testing_cluster) + len(e.testing_cluster))]
		points = np.concatenate([c.testing_cluster, d.testing_cluster, e.testing_cluster])

		for i, point in enumerate(points):
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
				boundary[i] = 1
			elif min(c_dist, d_dist, e_dist) == d_dist:
				boundary[i] = 2
			else:
				boundary[i] = 3

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating KNN3 error... Row: {0:4}/{1:4}'.format(i + 1, len(boundary)))

		# Confusion Matrix for KNN3
		c_matrix = confusion_matrix(testing_points_cde, boundary)

		# Calculate Error Rate for KNN3
		error_rate = 1 - (accuracy_score(testing_points_cde, boundary, normalize=True))  # error rate = 1 - accuracy score

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return c_matrix, error_rate
