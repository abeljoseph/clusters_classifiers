import numpy as np
import matplotlib.pyplot as plt
import sys

from math import pi, sqrt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class class_:
	def __init__(self, n, mean, covariance):
		self.covariance = covariance
		self.mean = mean
		self.n = n
		self.cluster = self.create_normal_distribution()
		self.eigenvals, self.eigenvecs = np.linalg.eig(self.covariance)
		self.testing_cluster = self.create_normal_distribution()
		

	def create_normal_distribution(self):
		return np.random.multivariate_normal(self.mean, self.covariance, size=self.n)


	def plot(self, ax):
		max_index = np.where(self.eigenvals == max(self.eigenvals))[0][0]
		min_index = np.where(self.eigenvals == min(self.eigenvals))[0][0]
		largest_eigval = self.eigenvals[max_index]
		smallest_eigval = self.eigenvals[min_index]
		largest_eigvec = self.eigenvecs[:, max_index]

		theta = np.arctan2(*largest_eigvec[::-1])
		theta_grid = np.linspace(0, 2 * pi)

		dim_a = sqrt(largest_eigval)
		dim_b = sqrt(smallest_eigval)

		axes_x = dim_a * np.cos(theta_grid)
		axes_y = dim_b * np.sin(theta_grid)

		rtn = [[np.cos(theta), np.sin(theta)], [-1 * np.sin(theta), np.cos(theta)]]

		ellipse = np.matmul(np.array([axes_x, axes_y]).T, rtn)
		ax.plot([x[0] + self.mean[0] for x in ellipse], [x[1] + self.mean[1] for x in ellipse])
		ax.scatter([x[0] for x in self.cluster], [x[1] for x in self.cluster])


class classifier:
	@staticmethod
	def get_micd_dist(obj, coord):
		return sqrt(np.matmul(np.matmul(np.subtract(coord, obj.mean), np.linalg.inv(obj.covariance)),
							  np.subtract(coord, obj.mean).T))


	@staticmethod
	def get_euclidean_dist(px1, py1, px0, py0):
		return sqrt((px0 - px1)**2 + (py0 - py1)**2)


	@staticmethod
	def create_med2(a, b):
		num_steps = 500

		# Create Mesh grid
		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1,
							 num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1,
							 num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]
		med2_cm_boundary =[0 for _ in range(len(a.cluster) + len(b.cluster))]

		for i in range(num_steps):
			for j in range(num_steps):
				a_dist = classifier.get_euclidean_dist(a.mean[0], a.mean[1], x0[i][j], y0[i][j])
				b_dist = classifier.get_euclidean_dist(b.mean[0], b.mean[1], x0[i][j], y0[i][j])

				boundary[i][j] = a_dist - b_dist
			
			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating MED2... Row: {0:4}/{1:4}'.format(i + 1, num_steps))
		
		points = np.concatenate([a.cluster, b.cluster])
		for i in range(len(points)):
			a_dist = sqrt((points[i][0] - a.mean[0])**2 + (points[i][1] - a.mean[1])**2)
			b_dist = sqrt((points[i][0] - b.mean[0])**2 + (points[i][1] - b.mean[1])**2)

			if min(a_dist, b_dist) == a_dist:
				med2_cm_boundary[i] = 1
			else:
				med2_cm_boundary[i] = 2

		print('... completed.')
		return [boundary, med2_cm_boundary, x_grid, y_grid]


	@staticmethod
	def create_med3(c, d, e):
		num_steps = 500

		# Create Mesh grid
		x_grid = np.linspace(min(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) - 1,
							 max(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) - 1,
							 max(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]
		med3_cm_boundary =[0 for _ in range(len(c.cluster) + len(d.cluster) + len(e.cluster))]

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

		print('... completed.')
		return [boundary, med3_cm_boundary, x_grid, y_grid]


	@staticmethod
	def create_ged2(a, b):
		num_steps = 500

		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1,
							 num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1,
							 num_steps)

		x, y = np.meshgrid(x_grid, y_grid)

		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]
		ged2_cm_boundary =[0 for _ in range(len(a.cluster) + len(b.cluster))]

		for i in range(num_steps):
			for j in range(num_steps):
				coord = [x[i][j], y[i][j]]
				a_dist = classifier.get_micd_dist(a, coord)
				b_dist = classifier.get_micd_dist(b, coord)
				boundary[i][j] = (a_dist - b_dist)
			
			# Print progress
			sys.stdout.write('\r')
			sys.stdout.write('Calculating GED2... Row: {0:4}/{1:4}'.format(i + 1, num_steps))
		
		points_ab = np.concatenate([a.cluster, b.cluster])
		for i,point in enumerate(points_ab):
			a_dist = classifier.get_micd_dist(a, point)
			b_dist = classifier.get_micd_dist(b, point)

			if min(a_dist, b_dist) == a_dist:
				ged2_cm_boundary[i] = 1
			else:
				ged2_cm_boundary[i] = 2

		print('... completed.')
		return [boundary, ged2_cm_boundary, x_grid, y_grid]


	@staticmethod
	def create_ged3(c, d, e):
		num_steps = 500

		x_grid = np.linspace(min(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) - 1,
							 max(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) - 1,
							 max(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) + 1, num_steps)

		x, y = np.meshgrid(x_grid, y_grid)

		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]
		ged3_cm_boundary = [0 for _ in range(len(c.cluster) + len(d.cluster) + len(e.cluster))]

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

		print('... completed.')
		return [boundary, ged3_cm_boundary, x_grid, y_grid]


	@staticmethod
	def create_map2(a, b):
		print('Calculating MAP2...', end=" ")
		num_steps = 100
		# Create Mesh grid
		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1,
							 num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1,
							 num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		# Calculate P(a) and P(b)
		p_a = a.n / (a.n + b.n)
		p_b = b.n / (a.n + b.n)

		# Calculate marginal of a and b


	@staticmethod
	def create_map3(c, d, e):
		print('Calculating MAP3...', end=" ")
		num_steps = 100

		# Create Mesh grid
		x_grid = np.linspace(min(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) - 1,
							 max(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) - 1,
							 max(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		# TODO: implement


	@staticmethod
	def create_nn2(a, b):
		num_steps = 100

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

		print('... completed.')
		return [boundary, x_grid, y_grid]
	

	@staticmethod
	def create_nn3(c, d, e):
		num_steps = 100

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

		print('... completed.')
		return [boundary, x_grid, y_grid]


	@staticmethod
	def create_knn2(a, b):
		num_steps = 100

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

		print('... completed.')
		return [boundary, x_grid, y_grid]
	

	@staticmethod
	def create_knn3(c, d, e):
		num_steps = 100

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
				sys.stdout.write('Calculating KNN2... Row: {0:4}/{1:4}'.format(i + 1, num_steps))

		print('... completed.')
		return [boundary, x_grid, y_grid]
	

	@staticmethod
	def nn2_test(a,b):
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


if __name__ == "__main__":
	# Instantiate classes
	a = class_(n=200, mean=[5, 10], covariance=[[8, 0], [0, 4]])
	b = class_(n=200, mean=[10, 15], covariance=[[8, 0], [0, 4]])
	c = class_(n=100, mean=[5, 10], covariance=[[8, 4], [4, 40]])
	d = class_(n=200, mean=[15, 10], covariance=[[8, 0], [0, 8]])
	e = class_(n=150, mean=[10, 5], covariance=[[10, -5], [-5, 20]])

	class_list = [a, b, c, d, e]

	############## --- Plot 1 --- ##############
	# Determine MED classifiers
	MED_ab, med2_cm_boundary, med_ab_x, med_ab_y = classifier.create_med2(a, b)
	MED_cde, med3_cm_boundary, med_cde_x, med_cde_y = classifier.create_med3(c, d, e)

	# Determine GED classifiers
	GED_ab, ged2_cm_boundary, ged_ab_x, ged_ab_y = classifier.create_ged2(a, b)
	GED_cde, ged3_cm_boundary, ged_cde_x, ged_cde_y = classifier.create_ged3(c, d, e)

	# Create scatters and set appearance for MED, GED, and MAP
	fig1, axs1 = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'aspect': 1})

	for ax in axs1:
		ax.set(xlabel='Feature 1', ylabel='Feature 2')
		ax.set_aspect('equal')
		ax.grid()

	# Plot A and B
	axs1[0].set_title("Feature 2 vs. Feature 1 for classes A and B")
	a.plot(axs1[0])
	b.plot(axs1[0])

	# Plot Classifiers
	axs1[0].contour(med_ab_x, med_ab_y, MED_ab, levels=[0], colors="black")
	axs1[0].contour(ged_ab_x, ged_ab_y, GED_ab, levels=[0], colors="red")
	axs1[0].legend(["Class A", "Class B"])

	# Plot C, D, E
	axs1[1].set_title("Feature 2 vs. Feature 1 for classes C, D and E")
	c.plot(axs1[1])
	d.plot(axs1[1])
	e.plot(axs1[1])

	# Plot Classifiers
	axs1[1].contour(med_cde_x, med_cde_y, MED_cde, colors="black")
	axs1[1].contour(ged_cde_x, ged_cde_y, GED_cde, colors="red")
	axs1[1].legend(["Class C", "Class D", "Class E"])

	############## --- Plot 2 --- ##############
	# Determine NN classifiers
	NN_ab, nn_ab_x, nn_ab_y = classifier.create_nn2(a, b)
	NN_cde, nn_cde_x, nn_cde_y = classifier.create_nn3(c, d, e)

	# Determine KNN classifiers
	KNN_ab, knn_ab_x, knn_ab_y = classifier.create_knn2(a, b)
	KNN_cde, knn_cde_x, knn_cde_y = classifier.create_knn3(c, d, e)

	# Create scatters and set appearance for NN, and 5NN
	fig2, axs2 = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'aspect': 1})

	for ax in axs2:
		ax.set(xlabel='Feature 1', ylabel='Feature 2')
		ax.set_aspect('equal')
		ax.grid()

	# Plot A and B
	axs2[0].set_title("Feature 2 vs. Feature 1 for classes A and B")
	a.plot(axs2[0])
	b.plot(axs2[0])

	# Plot Classifiers
	axs2[0].contour(nn_ab_x, nn_ab_y, NN_ab, levels=[0], colors="red")
	axs2[0].contour(knn_ab_x, knn_ab_y, KNN_ab, levels=[0], colors="black")
	axs2[0].legend(["Class A", "Class B"])

	# Plot C, D, E
	axs2[1].set_title("Feature 2 vs. Feature 1 for classes C, D and E")
	c.plot(axs2[1])
	d.plot(axs2[1])
	e.plot(axs2[1])

	# Plot Classifiers
	axs2[1].contour(nn_cde_x, nn_cde_y, NN_cde, colors="red")
	axs2[1].contour(knn_cde_x, knn_cde_y, KNN_cde, colors="black")
	axs2[1].legend(["Class C", "Class D", "Class E"])

	plt.show()

	########## --- Error Analysis --- ##########
	# Calculate errors
	nn2_cm_boundary = classifier.nn2_test(a, b)
	nn3_cm_boundary = classifier.nn3_test(c, d, e)
	knn2_cm_boundary = classifier.knn2_test(a, b)
	knn3_cm_boundary = classifier.knn3_test(c, d, e)
	
	# Making an array of 1s for all points in class A and an array of 2s for all points in class B
	# points contains all the actual values for class A and B
	points_a = [1 for x in a.cluster]
	points_b = [2 for x in b.cluster]
	points_ab = points_a + points_b

	# Making an array of 1s for all points in class C, array of 2s for all points in class D, array of 3s for class E
	# points contains all the actual values for class C, D, E
	points_c = [1 for x in c.cluster]
	points_d = [2 for x in d.cluster]
	points_e = [3 for x in e.cluster]
	points_cde = points_c + points_d + points_e

	# Confusion Matrix for MED2
	c_matrix_med2 = confusion_matrix(points_ab, med2_cm_boundary)
	print("Confusion Matrix MED2: \n {}".format(c_matrix_med2))

	# Calculate Error Rate for MED2
	med2_error_rate = 1 - (accuracy_score(points_ab, med2_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate MED2 = {}".format(med2_error_rate))

	# Confusion Matrix for MED3
	c_matrix_med3 = confusion_matrix(points_cde, med3_cm_boundary)
	print("Confusion Matrix MED3: \n {}".format(c_matrix_med3))

	# Calculate Error Rate for MED3
	med3_error_rate = 1 - (accuracy_score(points_cde, med3_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate MED3 = {}".format(med3_error_rate))

	# Confusion Matrix for GED2
	c_matrix_ged2 = confusion_matrix(points_ab, ged2_cm_boundary)
	print("Confusion Matrix GED2: \n {}".format(c_matrix_ged2))

	# Calculate Error Rate for GED2
	ged2_error_rate = 1 - (accuracy_score(points_ab, ged2_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate GED2 = {}".format(ged2_error_rate))

	# Confusion Matrix for GED3
	c_matrix_ged3 = confusion_matrix(points_cde, ged3_cm_boundary)
	print("Confusion Matrix GED3: \n {}".format(c_matrix_ged3))

	# Calculate Error Rate for GED3
	ged3_error_rate = 1 - (accuracy_score(points_cde, ged3_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate GED3 = {}".format(ged3_error_rate))

	# Confusion Matrix for NN2
	c_matrix_nn2 = confusion_matrix(points_ab, nn2_cm_boundary)
	print("Confusion Matrix NN2: \n {}".format(c_matrix_nn2))

	# Calculate Error Rate for NN2
	nn2_error_rate = 1 - (accuracy_score(points_ab, nn2_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate NN2 = {}".format(nn2_error_rate))

	# Confusion Matrix for NN3
	c_matrix_nn3 = confusion_matrix(points_cde, nn3_cm_boundary)
	print("Confusion Matrix NN3: \n {}".format(c_matrix_nn3))

	# Calculate Error Rate for NN3
	nn3_error_rate = 1 - (accuracy_score(points_cde, nn3_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate NN3 = {}".format(nn3_error_rate))

	# Confusion Matrix for KNN2
	c_matrix_knn2 = confusion_matrix(points_ab, knn2_cm_boundary)
	print("Confusion Matrix KNN2: \n {}".format(c_matrix_nn2))

	# Calculate Error Rate for KNN2
	knn2_error_rate = 1 - (accuracy_score(points_ab, knn2_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate KNN2 = {}".format(knn2_error_rate))

	# Confusion Matrix for KNN3
	c_matrix_knn3 = confusion_matrix(points_cde, knn3_cm_boundary)
	print("Confusion Matrix KNN3: \n {}".format(c_matrix_knn3))

	# Calculate Error Rate for KNN3
	knn3_error_rate = 1 - (accuracy_score(points_cde, knn3_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate KNN3 = {}".format(knn3_error_rate))

