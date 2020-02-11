import numpy as np
import matplotlib.pyplot as plt
import sys

from math import pi, sqrt


class class_:
	def __init__(self, n, mean, covariance):
		self.covariance = covariance
		self.mean = mean
		self.n = n
		self.cluster = []
		self.eigenvals = []
		self.eigenvecs = []


	@staticmethod
	def create_normal_distribution(size, mean, std_dev):
		return np.random.multivariate_normal(mean, std_dev, size=size)


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
		return sqrt(np.matmul(np.matmul(np.subtract(coord, obj.mean), np.linalg.inv(obj.covariance)), np.subtract(coord, obj.mean).T))


	@staticmethod
	def get_euclidean_dist(px1, py1, x0, y0, i, j):
		return sqrt((x0[i][j] - px1)**2 + (y0[i][j] - py1)**2)
			

	@staticmethod
	def create_med2(a, b):
		print('Calculating MED2...', end =" ")
		num_steps = 500

		# Create Mesh grid
		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary=[[0 for _ in range(len(x_grid))]for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				a_dist = classifier.get_euclidean_dist(a.mean[0], a.mean[1], x0, y0, i, j)
				b_dist = classifier.get_euclidean_dist(b.mean[0], b.mean[1], x0, y0, i, j)
				
				boundary[i][j] = a_dist - b_dist

		print('completed.')
		return [boundary, x_grid, y_grid]


	@staticmethod
	def create_med3(c, d, e):
		print('Calculating MED3...', end =" ")
		num_steps = 500

		# Create Mesh grid
		x_grid = np.linspace(min(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) - 1, max(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) - 1, max(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary=[[0 for _ in range(len(x_grid))]for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				c_dist = classifier.get_euclidean_dist(*c.mean, x0, y0, i, j)
				d_dist = classifier.get_euclidean_dist(*d.mean, x0, y0, i, j)
				e_dist = classifier.get_euclidean_dist(*e.mean, x0, y0, i, j)

				if min(c_dist, d_dist, e_dist) == c_dist:
					boundary[i][j] = 1
				elif min(c_dist, d_dist, e_dist) == d_dist:
					boundary[i][j] = 2
				else:
					boundary[i][j] = 3

		print('completed.')
		return [boundary, x_grid, y_grid]


	@staticmethod
	def create_ged2(a, b):
		print('Calculating GED2...', end =" ")
		num_steps = 500

		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1, num_steps)

		x, y = np.meshgrid(x_grid, y_grid)

		boundary=[[0 for _ in range(len(x_grid))]for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				coord = [x[i][j], y[i][j]]
				subtract_1 = classifier.get_micd_dist(a, coord)
				subtract_2 = classifier.get_micd_dist(b, coord)
				boundary[i][j] =  (subtract_1 - subtract_2)

		print('completed.')
		return [boundary, x_grid, y_grid]


	@staticmethod
	def create_ged3(c, d, e):
		print('Calculating GED3...', end =" ")
		num_steps = 500

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

		print('completed.')
		return [boundary, x_grid, y_grid]

	
	@staticmethod
	def create_nn2(a, b):
		print('Calculating NN2...')
		num_steps = 100

		# Create Mesh grid
		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary=[[0 for _ in range(len(x_grid))]for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				# Find nearest neighbours
				a_dist = float('inf')
				for coord in a.cluster:
					temp_dist = a_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0, y0, i, j)
					if temp_dist < a_dist:
						a_dist = temp_dist

				b_dist = float('inf')
				for coord in b.cluster:
					temp_dist = b_dist = classifier.get_euclidean_dist(coord[0], coord[1], x0, y0, i, j)
					if temp_dist < b_dist:
						b_dist = temp_dist
				
				boundary[i][j] = a_dist - b_dist

				# Print progress
				sys.stdout.write('\r')
				sys.stdout.write('{0:6.2f}% of {1:3}/{2:3}'.format((j+1)/num_steps*100, i+1, num_steps))

		print('... completed.')
		return [boundary, x_grid, y_grid]


if __name__ == "__main__":
	# Instantiate classes
	a = class_(n=200, mean=[5, 10], covariance=[[8, 0], [0, 4]])
	b = class_(n=200, mean=[10, 15], covariance=[[8, 0], [0, 4]])
	c = class_(n=100, mean=[5, 10], covariance=[[8, 4], [4, 40]])
	d = class_(n=200, mean=[15, 10], covariance=[[8, 0], [0, 8]])
	e = class_(n=150, mean=[10, 5], covariance=[[10, -5], [-5, 20]])

	class_list = [a, b, c, d, e]

	# Create clusters
	for cla in class_list:
		cla.cluster = class_.create_normal_distribution(cla.n, cla.mean, cla.covariance)

	# Determine eigenvalues
	for cla in class_list:
		cla.eigenvals, cla.eigenvecs = np.linalg.eig(cla.covariance)

	# Determine MED classifiers
	MED_ab, med_ab_x, med_ab_y = classifier.create_med2(a, b)
	MED_cde, med_cde_x, med_cde_y = classifier.create_med3(c, d, e)

	# Determine GED classifiers
	GED_ab, ged_ab_x, ged_ab_y = classifier.create_ged2(a, b)
	GED_cde, ged_cde_x, ged_cde_y = classifier.create_ged3(c, d, e)

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


	# Determine NN classifiers
	NN_ab, nn_ab_x, nn_ab_y = classifier.create_nn2(a, b)

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
	axs2[0].legend(["Class A", "Class B"])

	# Plot C, D, E
	axs2[1].set_title("Feature 2 vs. Feature 1 for classes C, D and E")
	c.plot(axs2[1])
	d.plot(axs2[1])
	e.plot(axs2[1])

	# Plot Classifiers
	axs2[1].legend(["Class C", "Class D", "Class E"])

	plt.show()