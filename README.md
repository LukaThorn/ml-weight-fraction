# ml-weight-fraction
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LukaThorn/ml-weight-fraction/master) 
[![GNUv3.0](https://img.shields.io/github/license/LukaThorn/ml-weight-fraction)](https://github.com/LukaThorn/ml-weight-fraction/blob/master/LICENSE)

This project uses machine learning (Python programming language) to approximate weight fractions of emerging chemicals in consumer products.

## Table of Contents
* [Launching the Project](#launching-the-project)
* [Core Components](#core-components)
* [Status](#status)
* [Contact](#contact)

## Launching the project
Before cloning this repository, try launching it through your browser:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LukaThorn/ml-weight-fraction/master)

Through the Binder interface, users may access all of our code in interactive iPython notebooks self-contained in a virtual, executable environment. I.e., the code for this project is entirely reproducible simply by clicking the button above.
### Suggested Software
If you prefer to clone our Git repository, we suggest these software versions:
* Python 3.7.3
* Poetry 1.0 or higher

After cloning, launch the project by running the following in your command line interface:
```
cd ml-weight-fraction
poetry install
poetry run jupyter lab
```
## Core Components
The suggested order to read/execute iPython notebooks:
* functions.ipynb: Contains functions used across multiple notebooks (optional with Binder).
* modelpipeline.ipynb: Framework for data-poor scenarios that tests and optimizes multiple machine learning prediction algorithms; includes (optional) augmentation of nanomaterials product data with organics product data.
* organicspipeline.ipynb: Framework for data-rich scenarios that tests and optimizes multiple machine learning prediction algorithms.

Post-processing data files are provided. If you wish to go through the pre-processing steps and generate data summaries, run these notebooks:
* preprocessENM.ipynb: Preprocessing steps for the nanomaterials product dataset (data-poor).
* preprocessorganics.ipynb: Preprocessing steps for the bulk-scale organic chemical product dataset (data-rich).
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
## Status
This project has been submitted for publication. Data sources are provided in the article.
## Contact
Luka L. Thornton (thorn.luka@gmail.com)
