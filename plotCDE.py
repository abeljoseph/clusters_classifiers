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


c = class_(n=100, mean=np.array([5, 10]), covariance=np.array([[8, 4], [4, 40]]))
d = class_(n=200, mean=[15, 10], covariance=[[8, 0], [0, 8]])
e = class_(n=150, mean=[10, 5], covariance=[[10, -5], [-5, 20]])

c.cluster = class_.create_normal_distribution(c.n, c.mean, c.covariance)
d.cluster = class_.create_normal_distribution(d.n, d.mean, d.covariance)
e.cluster = class_.create_normal_distribution(e.n, e.mean, e.covariance)
