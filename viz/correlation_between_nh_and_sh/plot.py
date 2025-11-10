#mcandrew

import sys

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

import pandas as pd
import matplotlib.pyplot as plt

import scienceplots
import seaborn as sns

import math
from datetime import datetime
from epiweeks import Week

from matplotlib.gridspec import GridSpec

def fit_sinusoidal_regression(y_data, print_results=True):
    """
    Fit a sinusoidal regression y = A*sin(2*pi*(t-phi)/52) + C to the input data.
    
    Parameters:
    -----------
    y_data : array-like
        The dependent variable data to fit
    print_results : bool, optional
        Whether to print the fitted parameters (default: True)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'A': amplitude parameter
        - 'phi': phase shift parameter
        - 'C': intercept parameter
        - 'success': boolean indicating if fitting was successful
        - 't_fitted': time points for plotting fitted curve
        - 'y_fitted': fitted y values for plotting
    """
    
    # Create sequential time index
    t_data = np.arange(len(y_data))
    
    # Define sinusoidal function: y = A*sin(2*pi*(t-phi)/52) + C
    def sinusoidal_func(t, A, phi, C):
        return A * np.sin(2 * np.pi * (t - phi) / 52) + C
    
    # Fit sinusoidal regression
    try:
        # Initial parameter guesses: A = amplitude estimate, phi = phase shift estimate, C = mean
        initial_guess = [np.std(y_data), 0, np.mean(y_data)]
        popt, pcov = curve_fit(sinusoidal_func, t_data, y_data, p0=initial_guess)
        A_fitted, phi_fitted, C_fitted = popt
        
        # Generate fitted curve for plotting
        t_fitted = np.linspace(0, len(y_data)-1, 100)
        y_fitted = sinusoidal_func(t_fitted, A_fitted, phi_fitted, C_fitted)
        
        # Print fitted parameters if requested
        if print_results:
            print(f"\nSinusoidal Regression Results:")
            print(f"A (amplitude) = {A_fitted:.6f}")
            print(f"phi (phase shift) = {phi_fitted:.6f}")
            print(f"C (intercept) = {C_fitted:.6f}")
            print(f"Function: y = {A_fitted:.6f} * sin(2*pi*(t - {phi_fitted:.6f})/52) + {C_fitted:.6f}")
        
        return {
            'A': A_fitted,
            'phi': phi_fitted,
            'C': C_fitted,
            'success': True,
            't_fitted': t_fitted,
            'y_fitted': y_fitted
        }
        
    except Exception as e:
        if print_results:
            print(f"\nSinusoidal regression fitting failed: {e}")
        
        return {
            'A': None,
            'phi': None,
            'C': None,
            'success': False,
            't_fitted': None,
            'y_fitted': None
        }

if __name__ == "__main__":

    colors = sns.color_palette("tab10", 5)
    plt.style.use("science")
    #fig,ax = plt.subplots()

    season_level = pd.read_csv("./analysis_data/season_level_data.csv")

    NH = season_level.loc[season_level.HEMISPHERE=="NH"]
    SH = season_level.loc[season_level.HEMISPHERE=="SH"]

    SH = SH.loc[SH.SEASON<=2024]


    fig = plt.figure(layout="constrained")

    gs = GridSpec(2, 2, figure=fig)


    ax = fig.add_subplot(gs[:,0])
    
    D  = np.vstack([SH.P.values, NH.P.values  ]).T
    mu = D.mean(0)
    C  = (D - mu)
    C  = C.T.dot(C) / (len(C)-1)
    mvn = stats.multivariate_normal( mu, C )
 
    ds          = np.arange(0.0,0.40+0.01,0.01)
    meshx,meshy = np.meshgrid( ds,ds )
    pos         = np.dstack((meshx, meshy)) 
    Z           = mvn.pdf(pos)    
    
    ax.contourf( meshx,meshy,Z, cmap = "Purples")

    ax.scatter(SH.P.values,NH.P.values,facecolors='none', edgecolors='black')

    b1,b0   = np.polyfit(SH.P.values    ,NH.P.values    ,1)
    b12,b02 = np.polyfit(SH.P.values[1:],NH.P.values[1:],1)

   
    x0,x1 = ax.get_xlim()
    ax.plot( [x0,x1], [b0+x0*b1,b0+x1*b1], color = colors[2],ls="--"  )
    #ax.plot( [x0,x1], [b02+x0*b12,b02+x1*b12], color = colors[3],ls="-."  )

    spear  = stats.spearmanr( SH.P.values, NH.P.values )
    linear = stats.pearsonr( SH.P.values, NH.P.values )
    linear_est, linear_p = linear
    linear_l,linear_u = linear.confidence_interval()
    
    spear_est, spear_p = spear
    stderr   = 1.0 / math.sqrt(len(SH) - 3)
    delta    = 1.96 * stderr
    spear_l  = math.tanh(math.atanh(spear_est) - delta)
    spear_u  = math.tanh(math.atanh(spear_est) + delta)
    
    ax.text( 0.98, 0.25, s = "Linear: {:.2f} [{:.2f}, {:.2f}]\nSpearman: {:.2f} [{:.2f}, {:.2f}]\np $<$ 0.01 for both".format(linear_est
                                                                                                                                    ,linear_l
                                                                                                                                    ,linear_u
                                                                                                                                    ,spear_est
                                                                                                                                    ,spear_l
                                                                                                                                    ,spear_u)
                                                                                                                                    ,ha="right",va="top", transform=ax.transAxes, fontsize=9)
   

    ax.set_xlabel("South Hem. \% positive",labelpad=0.1)
    ax.set_ylabel("North Hem. \% positive",labelpad=0.1)
    ax.set_xlim(0,0.4)
    ax.set_ylim(0,0.4)

    ax.set_xticks([0,.20,.40])
    ax.set_xticklabels(['0','20','40'])

    ax.set_yticks([0,.20,.40])
    ax.set_yticklabels(['0','20','40'])
    
    ax.text(0.95,0.95,s="A.", ha="right",va="top",transform=ax.transAxes)

    #time series
    d = pd.read_csv("./analysis_data/week_level_data.csv")
    d = d.loc[d.MMWRYR>=2009]

    before_2020 = d.loc[ (d.MMWRYR < 2020) ]
    after_2020  = d.loc[ (d.MMWRYR > 2020) ]

    colors = sns.color_palette("tab10", 5)

    #toprow = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs[0],wspace=0.05)
    
    #--TOPLEFT--
    ax__top_left = fig.add_subplot(gs[0,1])

    before_2020__NH = before_2020.loc[before_2020.HEMISPHERE=="NH"]
    before_2020__SH = before_2020.loc[before_2020.HEMISPHERE=="SH"]

    nh_line = ax__top_left.plot(before_2020__NH.MODELWEEK, before_2020__NH.POS.values, color = colors[0], lw=2,label="North Hem."  )
    twin =  ax__top_left.twinx()
    
    sh_line = twin.plot( before_2020__SH.MODELWEEK, before_2020__SH.POS.values, color = colors[1], lw=2,label="South Hem."  )

    # Move ax2 from right to left and offset
    twin.yaxis.set_label_position("left")
    twin.yaxis.tick_left()
    twin.spines["left"].set_position(("outward", 25))  # offset in points
    twin.spines["left"].set_visible(True)

    # Hide twin's right spine and ticks
    twin.spines["right"].set_visible(False)
    twin.yaxis.set_ticks_position('left')

    ax__top_left.set_yticks([0,25000,50000,95*10**3])
    ax__top_left.set_yticklabels(["0","25k","50k","95k"])
    ax__top_left.set_ylim(0,95*10**3)
    
    twin.set_yticks([0,2500,5000])
    twin.set_yticklabels(["0","2.5k","5.0k"])
    twin.set_ylim(0,6*10**3)

    twin.spines['left'].set_color(colors[1])         # Spine (line)
    twin.yaxis.label.set_color(colors[1])            # Axis label
    twin.tick_params(axis='y', colors=colors[1])     # Tick marks and tick labels

    twin.set_ylabel("Positive cases",color="black")

    xticks = [22,122,222,322,422,522]
    ax__top_left.set_xticks(xticks)
    twin.set_xticks(xticks)
    
    sub = before_2020__NH.loc[before_2020__NH.MODELWEEK.isin(xticks)]
    def mmwryw_to_month_year(mmwryw, is_first=False):
        year = int(str(mmwryw)[:4])
        week = int(str(mmwryw)[4:])
        mmwr_week = Week(year, week)
        date = mmwr_week.startdate()
        if is_first:
            return date.strftime("%m/%Y")
        else:
            return date.strftime("%m/%y")
    
    labels = []
    for i, (_, row) in enumerate(sub.iterrows()):
        is_first = (i == 0)
        labels.append(mmwryw_to_month_year(row.MMWRYW, is_first))
    ax__top_left.set_xticklabels(labels)

    ax__top_left.legend(handles=[nh_line[0], sh_line[0]], labels=["North Hem.", "South Hem."], frameon=False,loc="center")

    ax__top_left.text(1-0.95,0.95,s="B.", ha="left",va="top",transform=ax__top_left.transAxes)
 

    #--after 2020
    
    ax__top_right = fig.add_subplot(gs[1,1])
    
    after_2020__NH = after_2020.loc[after_2020.HEMISPHERE=="NH"]
    after_2020__SH = after_2020.loc[after_2020.HEMISPHERE=="SH"]
    
    # Fit sinusoidal regression to P column data
    regression_results_nh = fit_sinusoidal_regression(after_2020__NH['P'].values)
    regression_results_sh = fit_sinusoidal_regression(after_2020__SH['P'].values)

    ax__top_right.plot(after_2020__NH.MODELWEEK, after_2020__NH.POS.values, color = colors[0], lw=2  )
    
    # Add sinusoidal regression overlay for P column data if fitting was successful
    #if regression_results_nh['success']:
        #pass
        # Create a second y-axis for P values to overlay the sinusoidal fit
        # ax_p = ax__top_right.twinx()
        # ax_p.plot(after_2020__NH.MODELWEEK, after_2020__NH.P.values, color=colors[0], lw=2)
        
        # # Map fitted curve to MODELWEEK scale for plotting
        # t_data = np.arange(len(after_2020__NH['P'].values))
        # modelweek_fitted = np.interp(regression_results['t_fitted'], t_data, after_2020__NH.MODELWEEK.values)
        # ax_p.plot(modelweek_fitted, regression_results['y_fitted'], color='red', lw=2, linestyle='--', alpha=0.8, label='Sinusoidal fit')
        
        # ax_p.set_ylabel('P values', color='blue')
        # ax_p.tick_params(axis='y', labelcolor='blue')
        # ax_p.spines['right'].set_color('blue')
    
    twin =  ax__top_right.twinx()
    
    twin.plot( after_2020__SH.MODELWEEK, after_2020__SH.POS.values, color = colors[1], lw=2  )

    # Move ax2 from right to left and offset
    twin.yaxis.set_label_position("left")
    twin.yaxis.tick_left()
    twin.spines["left"].set_position(("outward", 25))  # offset in points
    twin.spines["left"].set_visible(True)

    # Hide twin's right spine and ticks
    twin.spines["right"].set_visible(False)
    twin.yaxis.set_ticks_position('left')

    ax__top_right.set_yticks([0,25000,50000,95*10**3])
    ax__top_right.set_yticklabels(["0","25k","50k","95k"])
    ax__top_right.set_ylim(0,95*10**3)
    
    twin.set_yticks([0,2500,5000])
    twin.set_yticklabels(["0","2.5k","5.0k"])
    twin.set_ylim(0,6*10**3)

    twin.spines['left'].set_color(colors[1])         # Spine (line)
    twin.yaxis.label.set_color(colors[1])            # Axis label
    twin.tick_params(axis='y', colors=colors[1])     # Tick marks and tick labels

    twin.set_ylabel("Positive cases",color="black")

    xticks = [626,726,826]
    ax__top_right.set_xticks(xticks)
    twin.set_xticks(xticks)
    
    sub = after_2020__NH.loc[after_2020__NH.MODELWEEK.isin(xticks)]
    ax__top_right.set_xticklabels( [ mmwryw_to_month_year(row.MMWRYW, is_first=False) for _,row in sub.iterrows()  ]  )

    ax__top_right.text(1-0.95,0.95,s="C.", ha="left",va="top",transform=ax__top_right.transAxes)

    fig.set_tight_layout(True)
    fig.set_size_inches(8.5-2,(11-2)/3 )
    fig.savefig("./viz/correlation_between_nh_and_sh/correlation_between_nh_and_sh.png",dpi=300)
    fig.savefig("./viz/correlation_between_nh_and_sh/correlation_between_nh_and_sh.pdf")

