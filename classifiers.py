import numpy as np
import sys
import time

from math import sqrt


class classifier:
	@staticmethod
	def get_micd_dist(obj, coord):
		return sqrt(np.matmul(np.matmul(np.subtract(coord, obj.mean), np.linalg.inv(obj.covariance)),
							  np.subtract(coord, obj.mean).T))

	@staticmethod
	def get_euclidean_dist(px1, py1, px0, py0):
		return sqrt((px0 - px1) ** 2 + (py0 - py1) ** 2)

	@staticmethod
	def create_med2(a, b):
		start_time = time.time()
		num_steps = 500

		# Create Mesh grid
		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1,
							 num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1,
							 num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				a_dist = classifier.get_euclidean_dist(a.mean[0], a.mean[1], x0[i][j], y0[i][j])
				b_dist = classifier.get_euclidean_dist(b.mean[0], b.mean[1], x0[i][j], y0[i][j])

				boundary[i][j] = a_dist - b_dist

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED2... Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return [boundary, x_grid, y_grid]

	@staticmethod
	def create_med3(c, d, e):
		start_time = time.time()
		num_steps = 500

		# Create Mesh grid
		x_grid = np.linspace(min(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) - 1,
							 max(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) - 1,
							 max(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				c_dist = classifier.get_euclidean_dist(*c.mean, x0[i][j], y0[i][j])
				d_dist = classifier.get_euclidean_dist(*d.mean, x0[i][j], y0[i][j])
				e_dist = classifier.get_euclidean_dist(*e.mean, x0[i][j], y0[i][j])

				if min(c_dist, d_dist, e_dist) == c_dist:
					boundary[i][j] = 1
				elif min(c_dist, d_dist, e_dist) == d_dist:
					boundary[i][j] = 2
				else:
					boundary[i][j] = 3
			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED3... Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return [boundary, x_grid, y_grid]

	@staticmethod
	def create_ged2(a, b):
		start_time = time.time()
		num_steps = 100

		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1,
							 num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1,
							 num_steps)

		x, y = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				coord = [x[i][j], y[i][j]]
				a_dist = classifier.get_micd_dist(a, coord)
				b_dist = classifier.get_micd_dist(b, coord)
				boundary[i][j] = (a_dist - b_dist)

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating GED2... Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return [boundary, x_grid, y_grid]

	@staticmethod
	def create_ged3(c, d, e):
		start_time = time.time()
		num_steps = 100

		x_grid = np.linspace(min(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) - 1,
							 max(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) - 1,
							 max(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) + 1, num_steps)

		x, y = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				coord = [x[i][j], y[i][j]]
				c_dist = classifier.get_micd_dist(c, coord)
				d_dist = classifier.get_micd_dist(d, coord)
				e_dist = classifier.get_micd_dist(e, coord)

				if min(c_dist, d_dist, e_dist) == c_dist:
					boundary[i][j] = 1
				elif min(c_dist, d_dist, e_dist) == d_dist:
					boundary[i][j] = 2
				else:
					boundary[i][j] = 3
			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating GED3... Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return [boundary, x_grid, y_grid]

	@staticmethod
	def get_map2(a, b, num_steps):

		# Create Mesh grid
		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1,
							 num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1,
							 num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		inv_cov_a = np.linalg.inv(a.covariance)
		inv_cov_b = np.linalg.inv(b.covariance)
		mean_a = np.array(a.mean)
		mean_b = np.array(b.mean)

		Q0 = np.subtract(inv_cov_a, inv_cov_b)
		Q1 = 2 * (np.dot(mean_b, inv_cov_b) - np.dot(mean_a, inv_cov_a))
		Q2 = np.dot(np.dot(mean_a, inv_cov_a), mean_a.T) - np.dot(np.dot(mean_b, inv_cov_b), mean_b.T)
		Q3 = np.log((b.n / a.n))
		Q4 = np.log(np.linalg.det(a.covariance) / np.linalg.det(b.covariance))

		for i in range(num_steps):
			for j in range(num_steps):
				coord = [x0[i][j], y0[i][j]]
				dist = np.matmul(np.matmul(coord, Q0), np.array(coord).T) + np.matmul(Q1, np.array(
					coord).T) + Q2 + 2 * Q3 + Q4
				boundary[i][j] = dist

		return [boundary, x_grid, y_grid]

	@staticmethod
	def create_map2(a, b):
		start_time = time.time()
		num_steps = 500

		boundary, x_grid, y_grid = classifier.get_map2(a, b, num_steps)

		for i in range(num_steps):
			for j in range(num_steps):
				boundary[i][j] = 1 if boundary[i][j] < 0 else 2

				# Print progress
				sys.stdout.write('\r')
				sys.stdout.write('Calculating MAP2... Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return [boundary, x_grid, y_grid]

	@staticmethod
	def create_map3(c, d, e):
		start_time = time.time()
		num_steps = 500

		# Create Mesh grid
		x_grid = np.linspace(min(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) - 1,
							 max(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) - 1,
							 max(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		boundary_cd = classifier.get_map2(c, d, num_steps)[0]
		boundary_ce = classifier.get_map2(c, e, num_steps)[0]
		boundary_de = classifier.get_map2(d, e, num_steps)[0]

		for i in range(num_steps):
			for j in range(num_steps):
				if boundary_cd[i][j] >= 0 and boundary_de[i][j] <= 0:
					boundary[i][j] = 2
				elif boundary_de[i][j] >= 0 and boundary_ce[i][j] >= 0:
					boundary[i][j] = 3
				elif boundary_ce[i][j] <= 0 and boundary_cd[i][j] <= 0:
					boundary[i][j] = 1

				# Print progress
				sys.stdout.write('\r')
				sys.stdout.write('Calculating MAP3... Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return [boundary, x_grid, y_grid]

	@staticmethod
	def create_nn2(a, b):
		start_time = time.time()
		num_steps = 50

		# Create Mesh grid
		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1,
							 num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1,
							 num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				# Find nearest neighbours
				a_dist = float('inf')
				for coord in a.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0[i][j], y0[i][j])
					if temp_dist < a_dist:
						a_dist = temp_dist

				b_dist = float('inf')
				for coord in b.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0[i][j], y0[i][j])
					if temp_dist < b_dist:
						b_dist = temp_dist

				boundary[i][j] = a_dist - b_dist

			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating NN2...  Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return [boundary, x_grid, y_grid]

	@staticmethod
	def create_nn3(c, d, e):
		start_time = time.time()
		num_steps = 50

		# Create Mesh grid
		x_grid = np.linspace(min(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) - 1,
							 max(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) - 1,
							 max(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				# Find nearest neighbours
				c_dist = float('inf')
				for coord in c.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0[i][j], y0[i][j])
					if temp_dist < c_dist:
						c_dist = temp_dist

				d_dist = float('inf')
				for coord in d.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0[i][j], y0[i][j])
					if temp_dist < d_dist:
						d_dist = temp_dist

				e_dist = float('inf')
				for coord in e.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0[i][j], y0[i][j])
					if temp_dist < e_dist:
						e_dist = temp_dist

				if min(c_dist, d_dist, e_dist) == c_dist:
					boundary[i][j] = 1
				elif min(c_dist, d_dist, e_dist) == d_dist:
					boundary[i][j] = 2
				else:
					boundary[i][j] = 3

				# Print progress
				sys.stdout.write('\r')
				sys.stdout.write('Calculating NN3...  Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return [boundary, x_grid, y_grid]

	@staticmethod
	def create_knn2(a, b):
		start_time = time.time()
		num_steps = 50

		# Create Mesh grid
		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1,
							 num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1,
							 num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				# Find nearest neighbours
				a_group = [float('inf') for _ in range(4)]
				for coord in a.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0[i][j], y0[i][j])
					if temp_dist < max(a_group):
						a_group[a_group.index(max(a_group))] = temp_dist

				a_dist = np.mean(a_group)

				b_group = [float('inf') for _ in range(4)]
				for coord in b.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0[i][j], y0[i][j])
					if temp_dist < max(b_group):
						b_group[b_group.index(max(b_group))] = temp_dist

				b_dist = np.mean(b_group)

				boundary[i][j] = a_dist - b_dist

				# Print progress
				sys.stdout.write('\r')
				sys.stdout.write('Calculating KNN2... Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return [boundary, x_grid, y_grid]

	@staticmethod
	def create_knn3(c, d, e):
		start_time = time.time()
		num_steps = 50

		# Create Mesh grid
		x_grid = np.linspace(min(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) - 1,
							 max(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) - 1,
							 max(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				# Find nearest neighbours
				c_group = [float('inf') for _ in range(4)]
				for coord in c.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0[i][j], y0[i][j])
					if temp_dist < max(c_group):
						c_group[c_group.index(max(c_group))] = temp_dist

				c_dist = np.mean(c_group)

				d_group = [float('inf') for _ in range(4)]
				for coord in d.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0[i][j], y0[i][j])
					if temp_dist < max(d_group):
						d_group[d_group.index(max(d_group))] = temp_dist

				d_dist = np.mean(d_group)

				e_group = [float('inf') for _ in range(4)]
				for coord in e.cluster:
					temp_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0[i][j], y0[i][j])
					if temp_dist < max(e_group):
						e_group[e_group.index(max(e_group))] = temp_dist

				e_dist = np.mean(e_group)

				if min(c_dist, d_dist, e_dist) == c_dist:
					boundary[i][j] = 1
				elif min(c_dist, d_dist, e_dist) == d_dist:
					boundary[i][j] = 2
				else:
					boundary[i][j] = 3

				# Print progress
				sys.stdout.write('\r')
				sys.stdout.write('Calculating KNN3... Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		end_time = time.time()
		print('... completed ({:9.4f} seconds).'.format(end_time - start_time))
		return [boundary, x_grid, y_grid]
