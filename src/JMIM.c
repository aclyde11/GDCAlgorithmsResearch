/*******************************************************************************
** MutualInformation.c
** Part of the mutual information toolbox
**
** Contains functions to calculate the mutual information of 
** two variables X and Y, I(X;Y), to calculate the joint mutual information
** of two variables X & Z on the variable Y, I(XZ;Y), and the conditional
** mutual information I(x;Y|Z)
** 
** Author: Adam Pocock
** Created 19/2/2010
** Updated - 22/02/2014 - Added checking on calloc.
**
** Copyright 2010-2017 Adam Pocock, The University Of Manchester
** www.cs.manchester.ac.uk
**
** This file is part of MIToolbox, licensed under the 3-clause BSD license.
*******************************************************************************/

#include "MIToolbox/JMIM.h"
#include "MIToolbox/MIToolbox.h"
#include "MIToolbox/MutualInformation.h"

double minJointMI(uint **feature_2d, uint *target, uint f_n, uint n, int *sset, uint k) {
	double min = 10000;
	double *temp1 = malloc(sizeof(double));
	double *temp2 = malloc(sizeof(double));
	int i; 

	for(i = 0; i < k; i++) {
		calcMutualInformations(feature_2d[sset[i]], target, n, temp1);
		calcConditionalMutualInformations(feature_2d[f_n], feature_2d[sset[i]], target, n, temp2);
		fprintf(stderr, "%f %f\n", *temp1, *temp2);
		*temp1 += *temp2;
		if(*temp1 < min) {
			min = *temp1;
		}
	}
	free(temp1);
	free(temp2);
	return min;
}

double joint_MI(uint *i, uint *j, uint *y, uint n){
	int count;
	 /*for(count = 0; count < 3; count++){
		fprintf(stderr, "%d %d %d \n", i[count], j[count], y[count]);
	} */
	double x = calcConditionalMutualInformation(i, y, j, n);
	double g = calcMutualInformation(j, y, n);
	return x;
}