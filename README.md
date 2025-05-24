# Post-Processing Approach for Distributive Fairness in Multi-Class Federated Learning

This repository contains the official implementation of the paper:

**Post-Processing Approach for Distributive Fairness in Multi-Class Federated Learning**  

## Overview

We propose a unified post-processing framework that enforces **distributive fairness** in **federated learning (FL)** across three dimensions:
- **Global group fairness** (e.g., fairness across legally protected attributes in the whole population)
- **Local group fairness** (e.g., fairness across legally protected attributes  within each client)
- **Client fairness** (e.g., Fairness across clients)

Our framework:
- Works in **multi-class FL** settings
- Supports common fairness metrics (e.g., Statistical Parity, Equal Opportunity)
- Is implemented as a **post-processing** linear program (LP) after model training
- Is computationally and communication-efficient

## Dataset
Adult: is used to predict whether an individual's income exceeds $50K/year based on census data. It contains 48,842 instances with 14 attributes.
Adult can be downloaded from: https://archive.ics.uci.edu/dataset/2/adult

PublicCoverage: Description: This dataset is derived from the American Community Survey (ACS) Public Use Microdata Sample (PUMS). It focuses on predicting whether an individual is covered by public health insurance.
It is detailed in: https://github.com/socialfoundations/folktables

HM1000: is a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The reference of HM 10000 is: https://www.nature.com/articles/sdata2018161
