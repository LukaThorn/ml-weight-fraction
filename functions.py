#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Readme" data-toc-modified-id="Readme-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Readme</a></span><ul class="toc-item"><li><span><a href="#Tips" data-toc-modified-id="Tips-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Tips</a></span></li></ul></li><li><span><a href="#Setup" data-toc-modified-id="Setup-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Setup</a></span></li><li><span><a href="#Classes" data-toc-modified-id="Classes-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Classes</a></span></li><li><span><a href="#Pre-processing-functions" data-toc-modified-id="Pre-processing-functions-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Pre-processing functions</a></span></li><li><span><a href="#Model-functions" data-toc-modified-id="Model-functions-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Model functions</a></span></li><li><span><a href="#Export" data-toc-modified-id="Export-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Export</a></span></li></ul></div>

# # Readme

# Source file for classes and functions.

# ## Tips

# * For writing error messages
#     * https://stackoverflow.com/questions/16451514/returning-error-string-from-a-function-in-python
# * List of objects: dir()

# # Setup

# In[1]:


from __future__ import division, print_function, unicode_literals
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import random
import pandas as pd
import os, sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[2]:


# Where to save the figures
PROJECT_ROOT_DIR = "."
PROJECT_SAVE_DIR = "figs"

if not (os.path.isdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)):
    print('Figure directory did not exist, creating now.')
    os.mkdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)
else:
    print('Figure directory exists.')


# In[3]:


# To enable a specified sound to play
from IPython.display import Audio
sound_file = './data/ping.wav'

# Option to play sound at the end of a function with a long run time
Audio(url=sound_file, autoplay=True)


# In[5]:


# Read in ENM feature data
X_enm = pd.read_csv("./data/ENM-preprocessed-feats.csv", sep='\t', 
                    header='infer', index_col=0)

# Read in ENM labels (maximum_weight_fraction)
y_enm = pd.read_csv("./data/ENM-clean.csv", sep=',', 
                    header='infer', usecols=[3])


# # Classes

# In[6]:


class HiddenPrints:
    """
    Option to suppress print output.
    Source:
    https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    """
    import os, sys
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# In[7]:


from sklearn.model_selection import GridSearchCV

class EstimatorSelectionHelper:
    """
    Set up grid search across multiple estimators, pipelines.
    By David Bastista:
    http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
    """
    #from sklearn.model_selection import GridSearchCV
    
    cv=10
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" 
                             % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=cv, n_jobs=1, verbose=1, 
            scoring='accuracy', refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


# # Pre-processing functions

# In[8]:


def bins(row):
    """
    Assign weight fractions (continuous) to bins (int).
    Class ranges are slightly different from those used by Isaacs et al. 2016.
    """
    if row['maximum_weight_fraction'] <= 0.002:
        val = 0 # low
    elif row['maximum_weight_fraction'] > 0.05:
        val = 2 # high
    else:
        val = 1 # medium
    return val


# In[9]:


bin_enm = np.asarray(y_enm.apply(bins, axis=1))


# In[10]:


def bar_graph_bins(label_data,
                   data_composition):
    """
    This function creates a bar graph of weight fraction bins and prints the 
    count and frequency for each.
    
    Arguments
    ----------
    label_data: int array of shape [n,]
        Dataframe containing binned wf data
    data_composition: string
        Describes the chemical composition of label_data 
        for use in the plot title; e.g., `ENM`, `Organics`   
    """
    import matplotlib.pyplot as plt
    
    # Find the count, frequency of WF bins
    unique, counts = np.unique(label_data, return_counts=True)
    wf_distrib = dict(zip(unique, counts))
    freq = []
    for i in counts:
        percent = (i/np.sum(counts)).round(2)
        freq.append(percent)

    # Plot
    plt.bar(range(len(wf_distrib)), list(wf_distrib.values()), align='center')
    plt.xticks(range(len(wf_distrib)), list(['low','medium','high']))
    plt.title('Frequency of %s Weight Fraction Bins' % data_composition)
    plt.show()
    
    print('Label bin: ', unique)
    print('Count    : ', counts)
    print('Frequency: ', freq)


# # Model functions

# In[17]:


def plot_conf_matrix(cm, 
                     classes, 
                     normalize=False, 
                     title='Confusion Matrix', 
                     cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Adapted from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    import matplotlib.pyplot as plt
    import itertools
    from sklearn.metrics import confusion_matrix
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel('True weight fraction')
    plt.xlabel('Predicted weight fraction')


# In[18]:


def plot_param_opt(param_grid, test_scores, scoring): 
    
    """
    Optional plot of validation score vs classifier parameter(s). For use 
    after running parameter optimization with GridSearchCV.
    """
    import matplotlib.pyplot as plt
    
    def convert_log_scale(n_set, n_label):
        log_dif = np.abs(np.log10(max(n_set)) - np.log10(min(n_set)))
        if log_dif > 3:
            n_set = np.log10(n_set)
            n_label = ('log_10(%s)' % n_label)    
        return n_set, n_label

    params = {k.split("__")[1]: v for k, v in param_grid.items()}
    param1_label = list(params.keys())[0]
    param1_set = list(params.values())[0]
    param1_set, param1_label = convert_log_scale(param1_set, param1_label)
    
    fig = plt.figure()
    if len(param_grid.keys()) == 1:
        plt.plot(param1_set, test_scores, 'k.-', ms=8, lw=2)
        plt.title('%s vs %s' % (scoring.title(), param1_label))
        plt.xlabel(param1_label)
        plt.ylabel(scoring.title())
        plt.xticks(np.arange(min(param1_set), max(param1_set) + 2, 2))
    elif len(param_grid.keys()) == 2:
        param2_label = list(params.keys())[1]
        param2_set = list(param_grid.values())[1]
        param2_set, param2_label = convert_log_scale(param2_set, param2_label)
        test_scores = np.reshape(test_scores, newshape=[-1, len(param2_set)])
        plt.contourf(param2_set, param1_set, test_scores)
        plt.title('%s Contours Over Parameter Grid' % scoring.title())
        plt.xlabel(param2_label)
        plt.ylabel(param1_label)
        plt.colorbar()
    plt.show()


# In[19]:


def plot_feat_impt(feature_names, 
                   importances, 
                   variances=None, 
                   save_fig_name=None,
                   combo_impt=False):
    """
    This function uses results from an rfc as input to plot feature importance.
    Here, the rfc determines importance using what is known as gini importance 
    or mean decrease impurity. Includes option to combine features into more 
    easily interpretable groups.
    
    References:
    https://stackoverflow.com/questions/15810339/how-are-feature-importances-in-randomforestclassifier-determined
    https://matplotlib.org/examples/api/barchart_demo.html
    https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
    """ 
    import matplotlib.pyplot as plt
    
    # (Optional) Sum importance by feature group
    if combo_impt:
        feature_names = ['chemProperties', 'functions', 'productCategories', 
                               'productType', 'productMatrix']
        importances = np.asarray([np.sum(importances[0:4]), 
                                  np.sum(importances[4:20]), 
                                  np.sum(importances[20:27]), 
                                  np.sum(importances[27:36]), 
                                  np.sum(importances[36:])])
        # (Optional) Sum variance by feature group
        if np.all(variances != None):
            variances = np.asarray([np.sum(variances[0:4]), 
                                    np.sum(variances[4:20]),
                                    np.sum(variances[20:27]),
                                    np.sum(variances[27:36]),
                                    np.sum(variances[36:])])
    
    indices = np.argsort(importances)
    
    # (Optional) Add error bars
    if np.all(variances != None):
        err_bars = np.sqrt(variances)
        fig, ax = plt.subplots()
        plt.grid(True)
        ax.barh(range(len(indices)), importances[indices], 
                 xerr=err_bars[indices], capsize=3, align='center')
    else: 
        fig, ax = plt.subplots()
        ax.barh(range(len(indices)), importances[indices], align='center')
    
    # Add grid lines
    plt.grid(False)
    ax.set_xticks(np.arange(0, np.amax(importances)+0.1, 0.05))
    ax.xaxis.grid(color='silver')
    ax.set_axisbelow(True)
    
    # Label parts of plot
    ax.set_title('Feature Importance')
    ax.set_xlabel('Relative Importance')
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    # Add importance value labels at the end of bars
    if variances is None:
        for rect in ax.patches:
            # Get X and Y placement of label from rect
            x_value = rect.get_width()
            y_value = rect.get_y() + rect.get_height() / 2
            # Use X value as label and format number with one decimal place
            label = "{:.2f}".format(x_value)
            # Create annotation
            plt.annotate(
                label,
                (x_value, y_value),         # Place label at end of the bar
                xytext=(4, 0),              # Horizontally shift label
                textcoords="offset points", # Interpret `xytext` as offset
                va='center', ha='left')
    
    fig = matplotlib.pyplot.gcf()
    if combo_impt: fig.set_size_inches(10, 6)
    else: fig.set_size_inches(10, 10)
    if np.all(save_fig_name != None):
        fig.savefig('./figs/feature_importance_%s.png' % save_fig_name, 
                   bbox_inches='tight')
    plt.show()


# In[22]:


# Define function to optimize, execute and evaluate a classifier using CV
from numpy import random

def model_opt_exe(classifier, 
                  X_training, 
                  y_training, 
                  X_testing=X_enm, 
                  y_testing=bin_enm, 
                  seed=random.randint(1,100),
                  save_fig_name=None, 
                  match_group=None, 
                  show_opt_plot=False, 
                  show_feat_impt=False, 
                  show_cnf_matrix=False, 
                  param_grid=None):
    """
    This function consists of three parts:
    1) Optimize the parameters for a classifier, either SVC-RBF or RFC;     
    2) Fit model pipeline to training data using optimized parameters and 
    stratified k-fold cross validation;
    3) Execute the optimized model and summarize its accuracy in a confusion 
    matrix broken down by WF bins. Formatted confusion matrices are saved as 
    .png files.
    
    Arguments
    ----------
    classifier: string ('svc' or 'rfc')
        The classifier to use in the pipeline; 'svc' refers to an SVC-RBF
    X_training: pandas data frame
        Feature data frame to train the model on
    y_training: pandas data frame
        WF (labels) data frame to train the model on
    X_testing: pandas data frame (default=X_enm)
        Feature data frame to test the best model on
    y_testing: pandas data frame (default=y_enm)
        WF (labels) data frame to test the best model on   
    seed: int (default=random.randint(1,100))
        Option to set the seed for CV
    save_fig_name: string (default=None)
        A unique string used at the end of confusion matrix and feature 
        importance (rfc-only) file names for exporting the figures as .png; 
        `None` indicates that no figures should be saved
    match_group: array of int (default=None)
        The array of ENM indices that augmented data were matched to; 
        applicable only to dfs with matching augmentation; prevents data leaks
    show_opt_plot: bool (default=False)
        `True` will plot accuracy as contour lines on the specified parameter 
        grid (svc) or a line plot of accuracy vs n_trees (rfc)
    show_cnf_matrix: bool (default=False)
        `True` results in matrix graphics being printed as output
    param_grid: dict (default=None)
        See param_grid for sklearn's GridSearchCV
    """     
    from sklearn.pipeline import Pipeline
    from sklearn import model_selection
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from numpy import random
    import matplotlib.pyplot as plt
    
    # =====PART 1=====
    # Optimize parameters
    
    # Define pipeline options for parameter optimization
    rfc = RandomForestClassifier(class_weight='balanced', 
                                 random_state=seed)
    svc = SVC(kernel='rbf', 
              class_weight='balanced',  # balances weights of WF bins
              random_state=seed)
    if classifier=='rfc':               # set pipeline for RFC
        pipe = Pipeline([
            ('scale', MinMaxScaler()),  # normalization from 0 to 1
            ('estimator', rfc)          # use RFC algorithm specified above
        ])
    else:                               # set pipeline for SVC-RBF
        pipe = Pipeline([
            ('scale', MinMaxScaler()),
            ('estimator', svc)
        ])

    # Set what kind of stratified k-fold CV to run
    num_folds = 10
    # When matching augmentation was NOT used, run normal stratified k-fold CV
    if np.all(match_group == None):
        cv = num_folds
    # When matching augmentation was used, keep each group of matched data 
    # samples together based on ENM index (match_group) when splitting data 
    # into folds so that there is no data leakage during CV
    else: 
        gfk = GroupKFold(n_splits=num_folds)
        gfk.random_state = seed
        cv = gfk.split(X_training, y_training, match_group)

    # Find best algorithm parameters by searching over a grid using the CV
    # and pipeline conditions specified above
    n_jobs = 3
    scoring = 'accuracy'
    grid_search = GridSearchCV(pipe, 
                               param_grid, 
                               cv=cv, 
                               scoring=scoring, 
                               n_jobs=n_jobs, 
                               pre_dispatch=2*n_jobs)
    grid_search.fit(X_training, y_training)
    
    # Retrieve accuracy scores for all grid search settings
    test_scores = grid_search.cv_results_.get('mean_test_score')
    
    # If optimization plotting is set as True, use plot_param_opt function
    # to plot a 2D or contour plot to visualize accuracy "hot spots"
    if show_opt_plot:
        plot_param_opt(param_grid, test_scores, scoring)
    
    # Retrieve best parameters from grid search (using list comprehension)
    best_params = {k.split("__")[1]: v 
                   for k, v in grid_search.best_params_.items()}
    
    # Print best accuracy and parameter values
    print('K-fold CV random state:\t', seed)
    print('Best fold %s:\t%.4f' % (scoring, grid_search.best_score_))
    for k, v in grid_search.best_params_.items(): 
        print('Best %s:\t%.2e' % (k, v))
    
    # =====PART 2=====
    # Fit optimized pipeline to training data
    
    # RFC pipeline                    
    if classifier == 'rfc':
        rfc = RandomForestClassifier(class_weight='balanced', 
                                     random_state=seed, 
                                     **best_params) # use optimized parameters
        pipe = Pipeline([
            ('scale', MinMaxScaler()),
            ('estimator', rfc)
        ])
        pipe.fit(X_training, y_training)        # fit pipeline to training data
        importances = rfc.feature_importances_  # get feature impt. from fit
        
        # Option to plot feature importance (RFC only)
        if show_feat_impt:
            feature_names = X_training.columns.values
            plot_feat_impt(feature_names, importances, save_fig_name)      
    
    # SVC pipeline
    else:
        svc = SVC(kernel='rbf', 
                  class_weight='balanced', 
                  random_state=seed, 
                  **best_params)                # use optimized parameters
        pipe = Pipeline([
            ('scale', MinMaxScaler()),
            ('estimator', svc)
        ])
        pipe.fit(X_training,y_training)
    
    # =====PART 3=====
    # Model execution and performance summary
    
    X = np.array(X_testing)
    y = np.array(y_testing)
    
    # Set CV as ~leave-one-out (based on sample size of the smallest WF bin)
    kfold = model_selection.StratifiedKFold(n_splits=17, # smallest bin size
                                            shuffle=True, 
                                            random_state=seed)
    
    # Placeholder matrix of accuracies averaged across CV folds
    cnf_matrix = np.zeros([3,3]) # 3 "true" vs 3 "predicted" WF bins
    
    # Run fitted pipeline using CV conditions defined above               
    for train_index, test_index in kfold.split(X,y):
        X_train, X_test = X[train_index], X[test_index] # split test data
        y_train, y_test = y[train_index], y[test_index] # into folds
        y_enm_predict = pipe.predict(X_test)
        y[test_index] = y_enm_predict
        # Write accuracy results to confusion matrix
        cnf_matrix += confusion_matrix(y_test, y_enm_predict)
    cnf_matrix = cnf_matrix.astype(np.int)
    np.set_printoptions(precision=2)
    class_names = ["low","mid","high"]

    # Plot and save non-normalized confusion matrix
    fig = plt.figure()
    plot_conf_matrix(cnf_matrix, classes=class_names, normalize=False)
    if np.all(save_fig_name != None):
        fig.savefig('./figs/confusion_notnorm_%s.png' % save_fig_name)
    if not show_cnf_matrix: plt.close(fig)

    # Plot and save normalized confusion matrix
    fig = plt.figure()
    plot_conf_matrix(cnf_matrix, classes=class_names, normalize=True,
                         title='Normalized Confusion Matrix')
    if np.all(save_fig_name != None):
        fig.savefig('./figs/confusion_norm_%s.png' % save_fig_name)
    if not show_cnf_matrix: plt.close(fig)
    
    # Calculate the average normalized accuracy across all bins
    cm_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:,np.newaxis]
    avg_norm = (cm_norm[0,0] + cm_norm[1,1] + cm_norm[2,2]) / 3
    print('Average normalized accuracy: ', avg_norm)
    
    # Play sound when done running
    display(Audio(url=sound_file, autoplay=True))
    
    # Set output based on chosen classifier
    if classifier == 'rfc':
        return avg_norm, importances
    else:
        return avg_norm


# In[23]:


def multi_trials(num_trials, 
                 model_params):
    """
    This function repeats model_opt_exe for a specified number of trials and
    provides summary statistics. Returns avg mean (scalar), avg stdev 
    (scalar), and optionally, for RFC, arrays for average feature importance 
    and variance.
    
    Arguments
    ----------
    num_trials: int
        The number of times to repeat
    model_params: dict
        A dictionary of parameters to run model_opt_exe 
    """  
    seed_set = np.random.choice(np.arange(1,101), 
                                size=num_trials, 
                                replace=False)
    with HiddenPrints():   # Hides function output for all the trials
        rs = []
        for seed in seed_set:
            model_params['seed'] = seed
            # Apply all-in-one function that optimizes and executes model
            rs_row = model_opt_exe(**model_params)
            rs.append(rs_row)
    # For RFC, write accuracy and feature importance results
    if model_params['classifier'] == 'rfc':
        results_accu = np.array([x for x, _ in rs]) # list comprehension
        results_impt = np.array([y for _, y in rs])
        avg_impt = results_impt.mean(axis=0)        # average importance
        var_impt = results_impt.var(axis=0)         # variance of importance
    # For SVC-RBF, only write accuracy results
    else:
        results_accu = np.array([x for x in rs])
       
    mu = results_accu.mean()   # average accuracy across trials
    sigma = results_accu.std() # standard deviation
    
    # Print summary statistics across trials
    print("Avg accuracy:    ", mu)
    print("Median accuracy: ", np.median(results_accu))
    print("StdDev accuracy: ", sigma)
    print("Numer of trials: ", num_trials)
    #print("Results: ", results_accu)
    
    # Play sound when done running
    display(Audio(url=sound_file, autoplay=True))
    
    # Set output based on chosen classifier
    if model_params['classifier'] == 'rfc':
        return mu, sigma, avg_impt, var_impt
    else: 
        return mu, sigma


# # Export

# In[24]:


get_ipython().system('jupyter nbconvert --to script functions.ipynb')


# In[ ]:




