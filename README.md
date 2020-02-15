# Clusters and Classifiers

This repository contains the source code for cluster generation and classification of five classes with bivariate Gaussian distribution parameters. Classification was implemented using MED, GED, MAP, NN, and kNN classifiers. Error analysis was conducted by determining the experimental error rate and confusion matrix for each classifier.  

![MED, GED and MAP classifiers](MED_GED_MAP.png)
![NN and kNN classifiers](NN_KNN.png)

## Cluster Generation
Normal distributions for the following classes were created using `numpy.random.multivariate_normal`. Plotted was implemented with `plot` and `scatter` from `matplotlib.pyplot`.  
Class A: N<sub>A</sub> = 200, μ<sub>A</sub> = [5 10]<sup>T</sup>,  Σ<sub>A</sub> = [ [8 0], [0 4] ]  
Class B: N<sub>B</sub> = 200, μ<sub>B</sub> = [10 15]<sup>T</sup>,  Σ<sub>B</sub> = [ [8 0], [0 4] ]  
  
Class C: N<sub>C</sub> = 100, μ<sub>C</sub> = [5 10]<sup>T</sup>,  Σ<sub>C</sub> = [ [8 4], [4 40] ]  
Class D: N<sub>D</sub> = 200, μ<sub>D</sub> = [15 10]<sup>T</sup>,  Σ<sub>D</sub> = [ [8 0], [0 8] ]  
Class E: N<sub>E</sub> = 150, μ<sub>D</sub> = [10 5]<sup>T</sup>,  Σ<sub>E</sub> = [ [10 -5], [-5 20] ]  


