#mcandrew

import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scienceplots

import seaborn as sns

if __name__ == "__main__":

    d = pd.read_csv("./analysis_data/week_level_data.csv")
    d = d.loc[d.MMWRYR>=2009]

    plt.style.use( "science" )
    
    fig = plt.figure()

    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2,1.25,2], hspace=0.65)

    before_2020 = d.loc[ (d.MMWRYR < 2020) ]
    after_2020  = d.loc[ (d.MMWRYR > 2020) ]

    colors = sns.color_palette("tab10", 5)

    toprow = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs[0],wspace=0.05)
    
    #--TOPLEFT--
    ax__top_left = fig.add_subplot(toprow[0])

    before_2020__NH = before_2020.loc[before_2020.HEMISPHERE=="NH"]
    before_2020__SH = before_2020.loc[before_2020.HEMISPHERE=="SH"]

    ax__top_left.plot(before_2020__NH.MODELWEEK, before_2020__NH.POS.values, color = colors[0], lw=2,label="North Hem."  )
    twin =  ax__top_left.twinx()
    
    twin.plot( before_2020__SH.MODELWEEK, before_2020__SH.POS.values, color = colors[1], lw=2,label="South Hem."  )

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
    ax__top_left.set_xticklabels( [ "{:02d}-{:02d}".format(int(str(row.MMWRYR)[-2:]),row.MMWRWK) for _,row in sub.iterrows()  ]  )

    ax__top_left.legend(frameon=False)

    ax__top_left.text(0.95,0.95,s="A.", ha="right",va="top",transform=ax__top_left.transAxes)
    
    
    #--TOPRIGHT--
    ax__top_right = fig.add_subplot(toprow[1])
    
    after_2020__NH = after_2020.loc[after_2020.HEMISPHERE=="NH"]
    after_2020__SH = after_2020.loc[after_2020.HEMISPHERE=="SH"]

    ax__top_right.plot(after_2020__NH.MODELWEEK, after_2020__NH.POS.values, color = colors[0], lw=2  )
    twin =  ax__top_right.twinx()
    
    twin.plot( after_2020__SH.MODELWEEK, after_2020__SH.POS.values, color = colors[1], lw=2  )

    #twin.set_visible(False)
    twin.get_yaxis().set_visible(False)     # hide ticks and labels
    twin.spines['left'].set_visible(False)  # or 'right' if you didn't move it
    twin.spines['right'].set_visible(False)

    ax__top_right.set_yticks([0,25000,50000])
    ax__top_right.set_yticklabels([])
    ax__top_right.set_ylim(0,95*10**3)
    
    twin.set_yticks([0,2500,5000])
    twin.set_yticklabels([])
    twin.set_ylim(0,6*10**3)

    xticks = [626,726,826]
    ax__top_right.set_xticks(xticks)
    twin.set_xticks(xticks)
    
    sub = after_2020__NH.loc[after_2020__NH.MODELWEEK.isin(xticks)]
    ax__top_right.set_xticklabels( [ "{:02d}-{:02d}".format(int(str(row.MMWRYR)[-2:]),row.MMWRWK) for _,row in sub.iterrows()  ]  )

    ax__top_right.text(0.95,0.95,s="B.", ha="right",va="top",transform=ax__top_right.transAxes)

    #--middle
    middle = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=gs[1], hspace=0.2 )   
    ax = fig.add_subplot(middle[0])
    
    NH = d.loc[d.HEMISPHERE=="NH"]
    SH = d.loc[d.HEMISPHERE=="SH"]
    
    def find_peak(x):
       return x.loc[ x.POS == np.max(x.POS)]
    NH_peaks = NH.groupby(["SEASON"]).apply(find_peak).reset_index(drop=True)
    SH_peaks = SH.groupby(["SEASON"]).apply(find_peak).reset_index(drop=True)

    NH_peaks = NH_peaks.loc[NH_peaks.SEASON>0]
    SH_peaks = SH_peaks.loc[SH_peaks.SEASON>0]

    ax.scatter( NH_peaks.SEASON, NH_peaks.MMWRWK,s=10,color=colors[0] )
    ax.scatter( SH_peaks.SEASON, SH_peaks.MMWRWK,s=10,color=colors[1] )

    ax.set_ylabel("Peak\nMMWR\nWeek")
    ax.set_xlabel("Influenza Season",labelpad=0.1)

    ax.text(0.95,0.95,s="C.", ha="right",va="top",transform=ax.transAxes)

    #--bottom
    bottomrow = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs[2],wspace=0.2)   
    
    ax = fig.add_subplot(bottomrow[0])

    season_level = pd.read_csv("./analysis_data/season_level_data.csv")

    NH = season_level.loc[season_level.HEMISPHERE=="NH"]
    SH = season_level.loc[season_level.HEMISPHERE=="SH"]

    SH = SH.loc[SH.SEASON<=2024]

    ax.scatter(SH.P.values,NH.P.values,facecolors='none', edgecolors='black')

    b1,b0 = np.polyfit(SH.P.values,NH.P.values,1)

    b12,b02 = np.polyfit(SH.P.values[1:],NH.P.values[1:],1)
    
    x0,x1 = ax.get_xlim()
    ax.plot( [x0,x1], [b0+x0*b1,b0+x1*b1], color = colors[2],ls="--"  )

    ax.plot( [x0,x1], [b02+x0*b12,b02+x1*b12], color = colors[3],ls="-."  )
    

    ax.set_xlabel("South Hem. \% pos.",labelpad=0.1)
    ax.set_ylabel("North Hem.\n\% pos.")
    ax.set_xlim(0,0.4)
    ax.set_ylim(0,0.4)

    ax.set_xticks([0,.20,.40])
    ax.set_xticklabels(['0','20','40'])

    ax.set_yticks([0,.20,.40])
    ax.set_yticklabels(['0','20','40'])
    
    ax.text(0.95,0.95,s="D.", ha="right",va="top",transform=ax.transAxes)

    ax = fig.add_subplot(bottomrow[1])

    weeks             = pickle.load(open("./viz/showcase_data_and_idea/weeks.pkl","rb"))
    weekly_infections = pickle.load(open("./viz/showcase_data_and_idea/weekly_infections.pkl","rb"))
    
    ax.scatter(weeks, weekly_infections,s=8, color="black")
    ax.set_xlabel("MMWR week"     )
    ax.set_ylabel("Incident cases",labelpad=0.5)

    ax.axvline(4,color="black",ls="--")

    control_infections = pickle.load(open("./viz/showcase_data_and_idea/control_infections.pkl","rb"))
    control_times      = pickle.load(open("./viz/showcase_data_and_idea/control_times.pkl"     ,"rb"))
 
    lower1,lower2,lower3,middle,upper3,upper2,upper1 = np.percentile(control_infections,[2.5, 10, 25,50,75,90,97.5],axis=0)
    ax.fill_between(weeks,lower1,upper1,alpha=0.2       ,color=colors[4])
    ax.fill_between(weeks,lower2,upper2,alpha=0.2       ,color=colors[4])
    ax.fill_between(weeks,lower2,upper2,alpha=0.2       ,color=colors[4])
    ax.plot(        weeks,middle                 ,lw=1.5, color=colors[4])

    ax.set_ylim(0,100)
    ax.set_yticks([0,50,100])

    ax.set_xticks([0,8,16,24,32])

    ax.text(0.95,0.95,s="E.", ha="right",va="top",transform=ax.transAxes)
    
    fig.set_size_inches( 7.5,10/3 )
    
    plt.savefig("./viz/showcase_data_and_idea/showcase.png", dpi=400)
    plt.savefig("./viz/showcase_data_and_idea/showcase.pdf", dpi=400)

    plt.close()
