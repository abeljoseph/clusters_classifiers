import numpy as np
import matplotlib.pyplot as plt

class class_:
	def __init__(self, n, mean, covariance):
		self.covariance = covariance
		self.mean = mean
		self.n = n
		self.cluster = 0

	@staticmethod
	def create_normal_distribution(size, mean, std_dev):
		return np.random.multivariate_normal(mean, std_dev, size=size)


a = class_(n=200, mean=[5, 10], covariance=[[8, 0], [0, 4]])
b = class_(n=200, mean=[10, 15], covariance=[[8, 0], [0, 4]])
c = class_(n=100, mean=[5, 10], covariance=[[8, 4], [4, 40]])
d = class_(n=200, mean=[15, 10], covariance=[[8, 0], [0, 8]])
e = class_(n=150, mean=[10, 5], covariance=[[10, -5], [-5, 20]])

a.cluster = class_.create_normal_distribution(a.n, a.mean, a.covariance)
b.cluster = class_.create_normal_distribution(b.n, b.mean, b.covariance)
c.cluster = class_.create_normal_distribution(c.n, c.mean, c.covariance)
d.cluster = class_.create_normal_distribution(d.n, d.mean, d.covariance)
e.cluster = class_.create_normal_distribution(e.n, e.mean, e.covariance)

plt.show()
