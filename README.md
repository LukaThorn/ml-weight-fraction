# ml-weight-fraction
This project uses machine learning (Python programming language) to approximate weight fractions of emerging chemicals in consumer products.

## Table of Contents
* [Intro](#intro)
* [Core Components](#core-components)
* [Status](#status)
* [Contact](#contact)

## Intro
### Suggested Software
* Python 3.7.3
* Poetry
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

## Core Components
The suggested order to read/execute code:
* functions.ipynb: Contains functions used across multiple notebooks.
* preprocessENM.ipynb: Preprocessing steps for the nanomaterials product dataset (data-poor).
* preprocessorganics.ipynb: Preprocessing steps for the bulk-scale organic chemical product dataset (data-rich).
* modelpipeline.ipynb: Framework for data-poor scenarios that tests and optimizes multiple machine learning prediction algorithms; includes (optional) augmentation of nanomaterials product data with organics product data.
* organicspipeline.ipynb: Framework for data-rich scenarios that tests and optimizes multiple machine learning prediction algorithms

## Status
This project has been submitted for publication.

## Contact
Luka L. Thornton (thorn.luka@gmail.com)
