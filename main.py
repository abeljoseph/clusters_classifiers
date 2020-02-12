import numpy as np
import matplotlib.pyplot as plt
import sys

from math import pi, sqrt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
	# Instantiate classes
	a = class_(n=200, mean=[5, 10], covariance=[[8, 0], [0, 4]])
	b = class_(n=200, mean=[10, 15], covariance=[[8, 0], [0, 4]])
	c = class_(n=100, mean=[5, 10], covariance=[[8, 4], [4, 40]])
	d = class_(n=200, mean=[15, 10], covariance=[[8, 0], [0, 8]])
	e = class_(n=150, mean=[10, 5], covariance=[[10, -5], [-5, 20]])

	class_list = [a, b, c, d, e]

	############## --- Plot 1 --- ##############
	print('---------------------- Plot 1 ----------------------')
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


	############## --- Plot 2 --- ##############
	print('\n---------------------- Plot 2 ----------------------')
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
	print('\n------------------ Error Analysis ------------------')
	
	# Calculate errors
	med2_cm_boundary = error_calc.med2_error(a, b)
	med3_cm_boundary = error_calc.med3_error(c, d, e)
	ged2_cm_boundary = error_calc.ged2_error(a, b)
	ged3_cm_boundary = error_calc.ged3_error(c, d, e)
	nn2_cm_boundary = error_calc.nn2_test(a, b)
	nn3_cm_boundary = error_calc.nn3_test(c, d, e)
	knn2_cm_boundary = error_calc.knn2_test(a, b)
	knn3_cm_boundary = error_calc.knn3_test(c, d, e)
	
	# Making an array of 1s for all points in class A and an array of 2s for all points in class B
	# points contains all the actual values for class A and B
	points_a = [1 for x in a.cluster]
	points_b = [2 for x in b.cluster]
	points_ab = points_a + points_b

	testing_points_a = [1 for x in a.testing_cluster]
	testing_points_b = [2 for x in b.testing_cluster]
	testing_points_ab = testing_points_a + testing_points_b

	# Making an array of 1s for all points in class C, array of 2s for all points in class D, array of 3s for class E
	# points contains all the actual values for class C, D, E
	points_c = [1 for x in c.cluster]
	points_d = [2 for x in d.cluster]
	points_e = [3 for x in e.cluster]
	points_cde = points_c + points_d + points_e

	testing_points_c = [1 for x in c.testing_cluster]
	testing_points_d = [2 for x in d.testing_cluster]
	testing_points_e = [3 for x in e.testing_cluster]
	testing_points_cde = testing_points_c + testing_points_d + testing_points_e

	# Confusion Matrix for MED2
	c_matrix_med2 = confusion_matrix(points_ab, med2_cm_boundary)
	print("\nConfusion Matrix MED2: \n {}".format(c_matrix_med2))

	# Calculate Error Rate for MED2
	med2_error_rate = 1 - (accuracy_score(points_ab, med2_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate MED2 = {}".format(med2_error_rate))

	# Confusion Matrix for MED3
	c_matrix_med3 = confusion_matrix(points_cde, med3_cm_boundary)
	print("\nConfusion Matrix MED3: \n {}".format(c_matrix_med3))

	# Calculate Error Rate for MED3
	med3_error_rate = 1 - (accuracy_score(points_cde, med3_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate MED3 = {}".format(med3_error_rate))

	# Confusion Matrix for GED2
	c_matrix_ged2 = confusion_matrix(points_ab, ged2_cm_boundary)
	print("\nConfusion Matrix GED2: \n {}".format(c_matrix_ged2))

	# Calculate Error Rate for GED2
	ged2_error_rate = 1 - (accuracy_score(points_ab, ged2_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate GED2 = {}".format(ged2_error_rate))

	# Confusion Matrix for GED3
	c_matrix_ged3 = confusion_matrix(points_cde, ged3_cm_boundary)
	print("\nConfusion Matrix GED3: \n {}".format(c_matrix_ged3))

	# Calculate Error Rate for GED3
	ged3_error_rate = 1 - (accuracy_score(points_cde, ged3_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate GED3 = {}".format(ged3_error_rate))

	# Confusion Matrix for NN2
	c_matrix_nn2 = confusion_matrix(testing_points_ab, nn2_cm_boundary)
	print("\nConfusion Matrix NN2: \n {}".format(c_matrix_nn2))

	# Calculate Error Rate for NN2
	nn2_error_rate = 1 - (accuracy_score(testing_points_ab, nn2_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate NN2 = {}".format(nn2_error_rate))

	# Confusion Matrix for NN3
	c_matrix_nn3 = confusion_matrix(testing_points_cde, nn3_cm_boundary)
	print("\nConfusion Matrix NN3: \n {}".format(c_matrix_nn3))

	# Calculate Error Rate for NN3
	nn3_error_rate = 1 - (accuracy_score(testing_points_cde, nn3_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate NN3 = {}".format(nn3_error_rate))

	# Confusion Matrix for KNN2
	c_matrix_knn2 = confusion_matrix(testing_points_ab, knn2_cm_boundary)
	print("\nConfusion Matrix KNN2: \n {}".format(c_matrix_nn2))

	# Calculate Error Rate for KNN2
	knn2_error_rate = 1 - (accuracy_score(testing_points_ab, knn2_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate KNN2 = {}".format(knn2_error_rate))

	# Confusion Matrix for KNN3
	c_matrix_knn3 = confusion_matrix(testing_points_cde, knn3_cm_boundary)
	print("\nConfusion Matrix KNN3: \n {}".format(c_matrix_knn3))

	# Calculate Error Rate for KNN3
	knn3_error_rate = 1 - (accuracy_score(testing_points_cde, knn3_cm_boundary, normalize=True)) #error rate = 1 - accuracy score
	print("Error Rate KNN3 = {}".format(knn3_error_rate))

