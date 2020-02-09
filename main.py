import numpy as np
import matplotlib.pyplot as plt

from math import pi


class class_:
	def __init__(self, n, mean, covariance):
		self.covariance = covariance
		self.mean = mean
		self.n = n
		self.cluster = 0
		self.eigenvals = []
		self.eigenvecs = []

	@staticmethod
	def create_normal_distribution(size, mean, std_dev):
		return np.random.multivariate_normal(mean, std_dev, size=size)


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

max_index = np.where(a.eigenvals == max(a.eigenvals))[0][0]
min_index = np.where(a.eigenvals == min(a.eigenvals))[0][0]
largest_eigval = a.eigenvals[max_index]
smallest_eigval = a.eigenvals[min_index]
largest_eigvec = a.eigenvecs[:, max_index]
smallest_eigvec = a.eigenvecs[:, min_index]

theta = np.arctan(largest_eigvec)
print(len(theta))
print(theta)

# Create scatters
# plt.scatter([x[0] for x in c.cluster], [x[1] for x in c.cluster])
# plt.scatter([x[0] for x in d.cluster], [x[1] for x in d.cluster])
# plt.scatter([x[0] for x in e.cluster], [x[1] for x in e.cluster])

# plt.show()
