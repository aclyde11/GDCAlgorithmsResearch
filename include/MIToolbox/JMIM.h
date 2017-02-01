/*******************************************************************************
** MutualInformation.h
** Part of the mutual information toolbox
**
** Contains functions to calculate the mutual information of 
** two variables X and Y, I(X;Y), to calculate the joint mutual information
** of two variables X & Z on the variable Y, I(XZ;Y), and the conditional
** mutual information I(x;Y|Z)
** 
** Author: Adam Pocock
** Created 19/2/2010
**
**  Copyright 2010-2017 Adam Pocock, The University Of Manchester
**  www.cs.manchester.ac.uk
**
**  This file is part of MIToolbox, licensed under the 3-clause BSD license.
*******************************************************************************/

#ifndef __MutualInformation_H
#define __MutualInformation_H

#include "MIToolbox/MIToolbox.h"
#include "MIToolbox/CalculateProbability.h"
#include "MIToolbox/MutualInformation.h"

#ifdef __cplusplus
extern "C" {
#endif 

double joint_MI(uint *i, uint *j, uint *target, uint n);
    double minJointMI(uint **feature_2d, uint *target, uint n, int *sset, uint k, uint f);

#ifdef __cplusplus
}
#endif

#endif

