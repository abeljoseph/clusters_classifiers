import matplotlib.pyplot as plt

from data_class import data_class_
from classifiers import classifier
from error_calculation import error_calc


if __name__ == "__main__":
	# Instantiate classes
	a = data_class_(n=200, mean=[5, 10], covariance=[[8, 0], [0, 4]])
	b = data_class_(n=200, mean=[10, 15], covariance=[[8, 0], [0, 4]])
	c = data_class_(n=100, mean=[5, 10], covariance=[[8, 4], [4, 40]])
	d = data_class_(n=200, mean=[15, 10], covariance=[[8, 0], [0, 8]])
	e = data_class_(n=150, mean=[10, 5], covariance=[[10, -5], [-5, 20]])

	class_list = [a, b, c, d, e]

	############## --- Plot 1 --- ##############
	print('---------------------- Plot 1 ----------------------')
	# Determine MED classifiers
	MED_ab, med_ab_x, med_ab_y = classifier.create_med2(a, b)
	MED_cde, med_cde_x, med_cde_y = classifier.create_med3(c, d, e)

	# Determine GED classifiers
	GED_ab, ged_ab_x, ged_ab_y = classifier.create_ged2(a, b)
	GED_cde, ged_cde_x, ged_cde_y = classifier.create_ged3(c, d, e)

	# Determine MAP classifiers
	# MAP_ab, map_ab_x, map_ab_y = classifier.create_map2(a, b)
	# MAP_cde, map_cde_x, map_cde_y = classifier.create_map3(c, d, e)

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
	#axs1[0].contour(map_ab_x, map_ab_y, MAP_ab, colors="green")
	axs1[0].legend(["Class A", "Class B"])

	# Plot C, D, E
	axs1[1].set_title("Feature 2 vs. Feature 1 for classes C, D and E")
	c.plot(axs1[1])
	d.plot(axs1[1])
	e.plot(axs1[1])

	# Plot Classifiers
	axs1[1].contour(med_cde_x, med_cde_y, MED_cde, colors="black")
	axs1[1].contour(ged_cde_x, ged_cde_y, GED_cde, colors="red")
	#axs1[1].contour(map_cde_x, map_cde_y, MAP_cde, colors="green")
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


	########## --- Error Analysis --- ##########
	print('\n------------------ Error Analysis ------------------')
	
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

	# Calculate confusion matrices and error rates
	c_matrix_med2, med2_error_rate = error_calc.med2_error(a, b, points_ab)
	c_matrix_med3, med3_error_rate = error_calc.med3_error(c, d, e, points_cde)
	c_matrix_ged2, ged2_error_rate = error_calc.ged2_error(a, b, points_ab)
	c_matrix_ged3, ged3_error_rate = error_calc.ged3_error(c, d, e, points_cde)
	#c_matrix_map2, map2_error_rate = error_calc.map2_error(a, b, points_ab)
	#c_matrix_map3, map3_error_rate = error_calc.map3_error(c, d, e, points_cde)
	c_matrix_nn2, nn2_error_rate = error_calc.nn2_test_error(a, b, testing_points_ab)
	c_matrix_nn3, nn3_error_rate = error_calc.nn3_test_error(c, d, e, testing_points_cde)
	c_matrix_knn2, knn2_error_rate = error_calc.knn2_test_error(a, b, testing_points_ab)
	c_matrix_knn3, knn3_error_rate = error_calc.knn3_test_error(c, d, e, testing_points_cde)

	# Print Confusion Matrix for MED2
	print("\nConfusion Matrix MED2: \n {}".format(c_matrix_med2))

	# Print Error Rate for MED2
	print("Error Rate MED2 = {:.3f}".format(med2_error_rate))

	# Print Confusion Matrix for MED3
	print("\nConfusion Matrix MED3: \n {}".format(c_matrix_med3))

	# Print Error Rate for MED3
	print("Error Rate MED3 = {:.3f}".format(med3_error_rate))

	# Print Confusion Matrix for GED2
	print("\nConfusion Matrix GED2: \n {}".format(c_matrix_ged2))

	# Print Error Rate for GED2
	print("Error Rate GED2 = {:.3f}".format(ged2_error_rate))

	# Print Confusion Matrix for GED3
	print("\nConfusion Matrix GED3: \n {}".format(c_matrix_ged3))

	# Print Error Rate for GED3
	print("Error Rate GED3 = {:.3f}".format(ged3_error_rate))

	# Print Confusion Matrix for MAP2
	#print("\nConfusion Matrix MAP2: \n {}".format(c_matrix_map2))

	# Print Error Rate for MAP2
	#print("Error Rate MAP2 = {:.3f}".format(map2_error_rate))

	# Print Confusion Matrix for MAP3
	#print("\nConfusion Matrix MAP3: \n {}".format(c_matrix_map3))

	# Print Error Rate for MAP3
	#print("Error Rate MAP3 = {:.3f}".format(map3_error_rate))

	# Print Confusion Matrix for NN2
	print("\nConfusion Matrix NN2: \n {}".format(c_matrix_nn2))

	# Print Error Rate for NN2
	print("Error Rate NN2 = {:.3f}".format(nn2_error_rate))

	# Print Confusion Matrix for NN3
	print("\nConfusion Matrix NN3: \n {}".format(c_matrix_nn3))

	# Print Error Rate for NN3
	print("Error Rate NN3 = {:.3f}".format(nn3_error_rate))

	# Print Confusion Matrix for KNN2
	print("\nConfusion Matrix KNN2: \n {}".format(c_matrix_nn2))

	# Print Error Rate for KNN2
	print("Error Rate KNN2 = {:.3f}".format(knn2_error_rate))

	# Print Confusion Matrix for KNN3
	print("\nConfusion Matrix KNN3: \n {}".format(c_matrix_knn3))

	# Print Error Rate for KNN3
	print("Error Rate KNN3 = {:.3f}".format(knn3_error_rate))


	# Show plots
	plt.show()