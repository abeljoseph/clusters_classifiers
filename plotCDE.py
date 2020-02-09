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
