import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib import patches

def read_errors_json(d):
    for i_cell, items in enumerate(d.items()):
        cell, entries_0 = items[0], items[1]
        for i_test, entries_1 in entries_0.items():
            if i_test == '0':
                err = pd.DataFrame(entries_1)
                err = err.unstack().to_frame().sort_index(level=1).T
                err.columns = err.columns.map('_'.join)
                data_i = err
                data_i.loc[:, "i_test"] = int(i_test)
                data_cell = data_i
            else:
                for iter, errs in enumerate(entries_1):
                    err = pd.DataFrame(errs)
                    err = err.unstack().to_frame().sort_index(level=1).T
                    err.columns = err.columns.map('_'.join)
                    if iter==0:
                        data_i = err
                    else:
                        data_i = pd.concat((data_i, err)).reset_index(drop=True)
                        
                data_i.loc[:, "i_test"] = int(i_test)
                data_cell = pd.concat((data_cell, data_i)).reset_index(drop=True)
        data_cell.loc[:, "Cell-Description"] = cell
        if i_cell==0:
            data_all = data_cell
        else:
            data_all = pd.concat((data_all, data_cell)).reset_index(drop=True)
    return data_all

def read_errors_soc(d):
    data_all = pd.DataFrame()
    for i_cell, items in enumerate(d.items()):
        cell, entries_0 = items[0], items[1]
        for SOC, entries_1 in entries_0.items():
            err = pd.DataFrame(entries_1)
            err = err.unstack().to_frame().sort_index(level=1).T
            err.columns = err.columns.map('_'.join)
            data_i = err
            data_i.loc[:, "SOC"] = int(SOC)
            data_cell = data_i
            
            data_cell.loc[:, "Cell-Description"] = cell
            data_all = pd.concat((data_all, data_cell)).reset_index(drop=True)
        
    return data_all

def read_predictions_json(d):
    targets = [
        'Total Heat Output [kJ/A*h]',
        'Cell Body Heat Output [kJ/A*h]',
        'Positive Heat Output [kJ/A*h]',
        'Negative Heat Output [kJ/A*h]',
    ]
    for i_cell, items in enumerate(d.items()):
        cell, entries_0 = items[0], items[1]
        for i_test, entries_1 in entries_0.items():
            if i_test == '0':
                data_i = pd.DataFrame(np.array(entries_1), columns=targets)
                data_i.loc[:, "i_test"] = int(i_test)
                data_i.loc[:, "iter"] = 0
                data_cell = data_i
            else:
                for iter, errs in enumerate(entries_1):
                    data_iter = pd.DataFrame(np.array(errs), columns=targets)
                    data_iter.loc[:, "iter"] = int(iter)
                    if iter==0:
                        data_i = data_iter
                    else:
                        data_i = pd.concat((data_i, data_iter)).reset_index(drop=True)
                        
                data_i.loc[:, "i_test"] = int(i_test)
                data_cell = pd.concat((data_cell, data_i)).reset_index(drop=True)
        data_cell.loc[:, "Cell-Description"] = cell
        if i_cell==0:
            data_all = data_cell
        else:
            data_all = pd.concat((data_all, data_cell)).reset_index(drop=True)
        data_all.loc[:, "Distribution"] = "Predicted"
    return data_all

def read_predictions_soc(d):
    data_all = pd.DataFrame()
    targets = [
        'Total Heat Output [kJ/A*h]',
        'Cell Body Heat Output [kJ/A*h]',
        'Positive Heat Output [kJ/A*h]',
        'Negative Heat Output [kJ/A*h]',
    ]
    for i_cell, items in enumerate(d.items()):
        cell, entries_0 = items[0], items[1]
#         print(cell, '+++++++++++++++++++++++')
        for SOC, entries_1 in entries_0.items():
            data_i = pd.DataFrame(np.array(entries_1), columns=targets)
            data_i.loc[:, "SOC"] = int(SOC)
            data_cell = data_i
            
            data_cell.loc[:, "Cell-Description"] = cell
            data_all = pd.concat((data_all, data_cell)).reset_index(drop=True)
            data_all.loc[:, "Distribution"] = "Predicted"

    return data_all

def plot_errors(errors, axes=None, linestyle='-', color='k', groupfunc='median', metric="rmse", label=None, y_upperlim=None):
    if groupfunc == 'mean':
        err_mean = errors.groupby(["Cell-Description","i_test"]).mean().reset_index()
    elif groupfunc == 'median':
        err_mean = errors.groupby(["Cell-Description","i_test"]).median().reset_index()
    err_lowCI_1sigma = errors.groupby(["Cell-Description","i_test"]).quantile(q=0.016).reset_index()
    err_lowCI_2sigma = errors.groupby(["Cell-Description","i_test"]).quantile(q=0.025).reset_index()
    err_highCI_1sigma = errors.groupby(["Cell-Description","i_test"]).quantile(q=0.84).reset_index()
    err_highCI_2sigma = errors.groupby(["Cell-Description","i_test"]).quantile(q=0.975).reset_index()
    cells = np.unique(errors['Cell-Description'])
    if np.all(axes==None):
        fig, axes = plt.subplots(2, 4, figsize=(16,4))
        axes = axes.ravel()
    
    ax_labels = ['a','b','c','d','e','f','g','h']
    metric_ = "total_" + metric
    idx = range(len(cells))
    for i, ax, cell, ax_label in zip(idx, axes, cells, ax_labels):
        err_mean_cell = err_mean.loc[err_mean["Cell-Description"] == cell, :]
        err_lowCI_1sig_cell = err_lowCI_1sigma.loc[err_lowCI_1sigma["Cell-Description"] == cell, :]
        err_highCI_1sig_cell = err_highCI_1sigma.loc[err_highCI_1sigma["Cell-Description"] == cell, :]
        err_lowCI_2sig_cell = err_lowCI_2sigma.loc[err_lowCI_2sigma["Cell-Description"] == cell, :]
        err_highCI_2sig_cell = err_highCI_2sigma.loc[err_highCI_2sigma["Cell-Description"] == cell, :]
        
        ax.plot(err_mean_cell["i_test"], err_mean_cell[metric_], linestyle, color=color, label=label)
        # ax.set_ylim(bottom=0, top=None)
        if metric != 'kl':
            if color=='b':
                ax.fill_between(err_mean_cell["i_test"], err_lowCI_1sig_cell[metric_], err_highCI_1sig_cell[metric_], alpha=0.4, color=color)
                ax.fill_between(err_mean_cell["i_test"], err_lowCI_2sig_cell[metric_], err_highCI_2sig_cell[metric_], alpha=0.2, color=color)
            else:
                ax.fill_between(err_mean_cell["i_test"], err_lowCI_1sig_cell[metric_], err_highCI_1sig_cell[metric_], alpha=0.2, color=color)
                ax.fill_between(err_mean_cell["i_test"], err_lowCI_2sig_cell[metric_], err_highCI_2sig_cell[metric_], alpha=0.1, color=color)
        ax.set_title(cell)
        if metric == 'kl':
            ax.set_yscale('log')
            ax.set_ylim([4e-4, 4])
        ax.text(0.03, 0.85, ax_label, fontsize=12, transform=ax.transAxes)
        if y_upperlim is not None:
            ax.set_ylim([0, y_upperlim[i]])
    
    axes[6].set_xticks([0, 3, 6, 9])
    axes[7].set_xticks([0, 3, 6, 9])
    if metric == 'kl':
        ylabel = 'Total Heat Output\nKL divergence'
    elif metric == 'rmse':
        ylabel = 'Total Heat Output\nRMSE [kJ/Ah]'

    axes[0].set_ylabel(ylabel)
    axes[4].set_ylabel(ylabel)
    axes[4].set_xlabel('Samples from test cell, i')
    axes[5].set_xlabel('Samples from test cell, i')
    axes[6].set_xlabel('Samples from test cell, i')
    axes[7].set_xlabel('Samples from test cell, i')
    axes[0].legend()
    plt.tight_layout()

def plot_errors_singleaxis(errors, ax=None, linestyle='-', color=None, groupfunc='median', metric="rmse", target="total"):
    if groupfunc == 'mean':
        err_mean = errors.groupby(["Cell-Description","i_test"]).mean().reset_index()
    elif groupfunc == 'median':
        err_mean = errors.groupby(["Cell-Description","i_test"]).median().reset_index()
    err_lowCI_1sigma = errors.groupby(["Cell-Description","i_test"]).quantile(q=0.016).reset_index()
    err_lowCI_2sigma = errors.groupby(["Cell-Description","i_test"]).quantile(q=0.025).reset_index()
    err_highCI_1sigma = errors.groupby(["Cell-Description","i_test"]).quantile(q=0.84).reset_index()
    err_highCI_2sigma = errors.groupby(["Cell-Description","i_test"]).quantile(q=0.975).reset_index()
    cells = pd.unique(errors['Cell-Description'])
    if np.all(ax==None):
        fig, ax = plt.subplots(1,1, figsize=(4,3.5))
    
    metric_ = target + "_" + metric
    for cell in cells:
        err_mean_cell = err_mean.loc[err_mean["Cell-Description"] == cell, :]
        err_lowCI_1sig_cell = err_lowCI_1sigma.loc[err_lowCI_1sigma["Cell-Description"] == cell, :]
        err_highCI_1sig_cell = err_highCI_1sigma.loc[err_highCI_1sigma["Cell-Description"] == cell, :]
        err_lowCI_2sig_cell = err_lowCI_2sigma.loc[err_lowCI_2sigma["Cell-Description"] == cell, :]
        err_highCI_2sig_cell = err_highCI_2sigma.loc[err_highCI_2sigma["Cell-Description"] == cell, :]
        if color is not None:
            line = ax.plot(err_mean_cell["i_test"], err_mean_cell[metric_], linestyle, color=color, label=cell)
        else:
            line = ax.plot(err_mean_cell["i_test"], err_mean_cell[metric_], linestyle, label=cell)
        ax.fill_between(err_mean_cell["i_test"], err_lowCI_2sig_cell[metric_], err_highCI_2sig_cell[metric_], alpha=0.3, color=line[0].get_color())
    
    if target == "total":
        target = "Total"
    elif target == "positive":
        target = "Positive"
    elif target == "body":
        target = "Cell Body"
    else:
        target = "Negative"
    
    if metric == 'kl':
        ylabel = target + " Heat Output\nKL divergence"
    elif metric == 'rmse':
        ylabel = target + " Heat Output\nRMSE [kJ/Ah]"

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Samples from test cell, i')
    ax.legend()
    plt.tight_layout()

    return ax

def calc_maha_depth(xy_to_plot):
    # xy_to_plot is numpy array with only x and y
    # Covariance matrix
    covariance  = np.cov(xy_to_plot , rowvar=False, bias=True)
    
        # Covariance matrix power of -1
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)
    
    # Center point########################################something better than the mean? #####
    centerpoint = np.mean(xy_to_plot , axis=0)
    
    # Distances between center point and 
    distances = []
    
    ###### CAN REDUCE THIS ####################################################################
    for i, val in enumerate(xy_to_plot):
        p1 = val
        p2 = centerpoint
        distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
        distances.append(distance)

    distances = np.array(distances)
    
    return covariance, distances, centerpoint
        
def mul_mahalanobis_plot(xy, other_xy, sz=100, ax=None, ellipse_n_prctiles=2):
    """Calculates and plots overlayed mahalanois plots for two xy sets"""
    # xy is numpy array
    covariance1, distances1, centerpoint1 = calc_maha_depth(xy)
    covariance2, distances2, centerpoint2 = calc_maha_depth(other_xy)
    
    ## Finding ellipse dimensions 
    pearson = covariance1[0, 1]/np.sqrt(covariance1[0, 0] * covariance1[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    lambda_, v = np.linalg.eig(covariance1)
    lambda_ = np.sqrt(lambda_)

    if ax is None:
        ax = plt.subplot()

    for i in range(1,ellipse_n_prctiles):
        i=i/ellipse_n_prctiles
        # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers 
        cutoff = chi2.ppf(i,xy.shape[1])

        # Index of outliers
        outlierIndexes = np.where(distances1 > cutoff)

        # Ellipse patch
        if i == 1:
            label='Actual Distribution'
        else:
            label='_'
        ellipse1 = patches.Ellipse(xy=(centerpoint1[0], centerpoint1[1]),
                        width=lambda_[0]*np.sqrt(cutoff)*2, height=lambda_[1]*np.sqrt(cutoff)*2,
                        angle=np.rad2deg(np.arccos(v[0, 0]) ), edgecolor='darkcyan', linewidth=3, label=label)
        ellipse1.set_facecolor('darkcyan') # #0984e3
        ellipse1.set_alpha(0.3)
        ellipse1.set_zorder(0)

        ax.add_artist(ellipse1)
    
    ## Finding ellipse dimensions 
    pearson = covariance2[0, 1]/np.sqrt(covariance2[0, 0] * covariance2[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    lambda_, v = np.linalg.eig(covariance2)
    lambda_ = np.sqrt(lambda_)

    for i in range(1,ellipse_n_prctiles):
        i=i/ellipse_n_prctiles
        # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers 
        cutoff = chi2.ppf(i,other_xy.shape[1])

        # Index of outliers
        outlierIndexes = np.where(distances2 > cutoff)

        # Ellipse patch
        if i == 1:
            label='Predicted Distribution'
        else:
            label='_'
        ellipse2 = patches.Ellipse(xy=(centerpoint2[0], centerpoint2[1]),
                        width=lambda_[0]*np.sqrt(cutoff)*2, height=lambda_[1]*np.sqrt(cutoff)*2,
                        angle=np.rad2deg(np.arccos(v[0, 0])), edgecolor='crimson', linestyle='--', linewidth=3, label=label)
        ellipse2.set_facecolor('crimson') # #0984e3
        ellipse2.set_alpha(0.3)
        ellipse2.set_zorder(0)

        ax.add_artist(ellipse2)

    ax.scatter(xy[: , 0], xy[ : , 1], c='cyan',edgecolor='black', zorder=1, s=sz, label="Actual Points")
    ax.scatter(other_xy[: , 0], other_xy[ : , 1], c='deeppink', marker="X",edgecolor='black', zorder=1, s=sz, label="Predicted Points")
    
    return ax