# ml-weight-fraction
This project uses machine learning (Python) to optimize an algorithm that can predict the weight fraction of emerging chemicals in consumer products.

## Table of Contents
* [Intro](#intro)
* [Features](#features)
* [Status](#status)
* [Contact](#contact)

## Intro
### Suggested Software
* Python 3.7.3 (via Anaconda v1.7.2)
* Jupyter Notebook (via Anaconda v1.7.2)
* R 3.6.0
* (Optional) R Studio
### Acronyms and Abbreviations
* arr = array
* bp = boiling point
* cv = cross validation
* df = data frame
* enm = engineered nanomaterials
* matrix_F = matrix of the product is a formulation (i.e., not a solid)
* ml = machine learning
* mp = melting point
* mw = molecular weight
* oecd = Organisation for Economic Co-operation and Development
* prop = property
* puc = product use category
* rbf = radial basis function (a non-linear implementation of SVM)
* rfc = random forest classifier
* svc = support vector classifier (for categorical data)
* svm = support vector machine
* wf = weight fraction

## Features
The suggested order to read/execute code:
* (optional) datacleaning.R: Assemble, merge and clean organics data. (to be uploaded)
* preprocessENM.ipynb: Preprocessing steps for ENM data, including feature agglomeration.
* preprocessorganics.ipynb: Preprocessing steps for organics data.
* modelpipeline.ipynb: Pipeline to test and optimize multiple machine learning algorithms; includes augmentation of ENM data with organics data.
* functions.ipynb: Contains functions used across multiple notebooks.

## Status
This project is still under development.

## Contact
B. Lila Thornton (bmt13@duke.edu)
