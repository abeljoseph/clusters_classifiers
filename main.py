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

	def plot(self):
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
		plt.plot([x[0] + self.mean[0] for x in ellipse], [x[1] + self.mean[1] for x in ellipse])
		plt.scatter([x[0] for x in self.cluster], [x[1] for x in self.cluster])


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

# Create scatters
a.plot()
b.plot()
e.plot()

plt.show()
