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

double minJointMI(uint **feature_2d, uint *target, uint samples, int *sset, uint k, uint f) {
    double min = 10000000, temp=0;
    for(int i = 0; i < k; i++) {
	temp =  joint_MI(feature_2d[f], feature_2d[sset[i]], target, samples);
	if(temp < min)
	    min = temp;
    }
    return min;
}

double joint_MI(uint *i, uint *j, uint *y, uint n){
    double x, g;
    calcConditionalMutualInformations(i, y, j, n, &x);
    calcMutualInformations(j, y, n, &g);
    return x + g;
}
