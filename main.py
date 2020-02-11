import numpy as np
import matplotlib.pyplot as plt

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
	def getEuclideanDistance(cla, x0, y0, i, j):
		return sqrt((x0[i][j] - cla.mean[0])**2 + (y0[i][j] - cla.mean[1])**2)
			
			
	@staticmethod
	def create_med2(a, b):
		num_steps = 1000

		# Create Mesh grid
		x_grid = np.linspace(min(*a.cluster[:, 0], *b.cluster[:, 0]) - 1, max(*a.cluster[:, 0], *b.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*a.cluster[:, 1], *b.cluster[:, 1]) - 1, max(*a.cluster[:, 1], *b.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary=[[0 for _ in range(len(x_grid))]for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				a_dist = classifier.getEuclideanDistance(a, x0, y0, i, j)
				b_dist = classifier.getEuclideanDistance(b, x0, y0, i, j)
				
				boundary[i][j] = a_dist - b_dist

		return [boundary, x_grid, y_grid]
		
	
	@staticmethod
	def create_med3(c, d, e):
		num_steps = 1000

		# Create Mesh grid
		x_grid = np.linspace(min(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) - 1, max(*c.cluster[:, 0], *d.cluster[:, 0], *e.cluster[:, 0]) + 1, num_steps)
		y_grid = np.linspace(min(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) - 1, max(*c.cluster[:, 1], *d.cluster[:, 1], *e.cluster[:, 1]) + 1, num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary=[[0 for _ in range(len(x_grid))]for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				c_dist = classifier.getEuclideanDistance(c, x0, y0, i, j)
				d_dist = classifier.getEuclideanDistance(d, x0, y0, i, j)
				e_dist = classifier.getEuclideanDistance(e, x0, y0, i, j)

				if min(c_dist, d_dist, e_dist) == c_dist:
					boundary[i][j] = 1
				elif min(c_dist, d_dist, e_dist) == d_dist:
					boundary[i][j] = 2
				else:
					boundary[i][j] = 3


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
	MED_ab, x_grid, y_grid = classifier.create_med2(a, b)
	MED_cde, x_grid1, y_grid1 = classifier.create_med3(c, d, e)

	# Create scatters and set appearance
	fig, axs = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'aspect': 1})
	
	for ax in axs:
		ax.set(xlabel='Feature 1', ylabel='Feature 2')
		ax.set_aspect('equal')
		ax.grid()
	
	# Plot A and B
	axs[0].set_title("Feature 2 vs. Feature 1 for classes A and B")
	a.plot(axs[0])
	b.plot(axs[0])

	# Plot Classifiers
	axs[0].contour(x_grid, y_grid, MED_ab, levels=[0], colors="black")
	
	axs[0].legend(["Class A", "Class B"])

	# Plot C, D, E
	axs[1].set_title("Feature 2 vs. Feature 1 for classes C, D and E")
	c.plot(axs[1])
	d.plot(axs[1])
	e.plot(axs[1])

	# Plot Classifiers
	axs[1].contour(x_grid1, y_grid1, MED_cde, colors="black")
	
	axs[1].legend(["Class C", "Class D", "Class E"])

	plt.show()
