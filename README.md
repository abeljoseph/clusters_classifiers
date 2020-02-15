# Clusters and Classifiers

This repository contains the source code for cluster generation and classification of five classes. The classes had bivariate Gaussian distribution parameters, and classification was implemented with MED, GED, MAP, NN, and kNN classifiers. Error analysis was conducted by determining the experimental error rate and confusion matrix for each classifier. 

## Cluster Generation
The following classes were plotted using `plot` and `contour` from `matplotlib.pyplot`.  
Class A: N<sub>A</sub> = 200, μ<sub>A</sub> = [5 10]<sup>T</sup>,  Σ<sub>A</sub> = [ [8 0], [0 4] ]  
Class B: N<sub>B</sub> = 200, μ<sub>B</sub> = [10 15]<sup>T</sup>,  Σ<sub>B</sub> = [ [8 0], [0 4] ]  
  
Class C: N<sub>C</sub> = 100, μ<sub>C</sub> = [5 10]<sup>T</sup>,  Σ<sub>C</sub> = [ [8 4], [4 40] ]  
Class D: N<sub>D</sub> = 200, μ<sub>D</sub> = [15 10]<sup>T</sup>,  Σ<sub>D</sub> = [ [8 0], [0 8] ]  
Class E: N<sub>E</sub> = 150, μ<sub>D</sub> = [10 5]<sup>T</sup>,  Σ<sub>E</sub> = [ [10 -5], [-5 20] ]  
