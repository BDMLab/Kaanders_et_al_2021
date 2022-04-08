import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import despine
import seaborn as sns
import matplotlib.patches as mpatches

sns.set(color_codes=True)

## PSD mod : To adjust our presentation I inverted the rating presentation from \ to left. HOwever, the variables are still labelled as left_minus_right 

def plot_fit(data, predictions, color_data = '#4F6A9A',label1 = 'Human',label2 = ' Simulations',legend_label = '', Inverse = False, GazePlot = True):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    sns.set(style='white', font_scale=1.5)
    plot_rt_by_difficulty_zSc(data, predictions,
                          xlims =(0, 50), xlabel_skip=2,color1 = color_data ,
                          ax=axs[0][0])
    plot_pleft_by_left_minus_mean_others(data, predictions,
                                         xlabel_skip=5, xlims=[-100, 100], xlabel_start=0,color1 = color_data, ax=axs[0][1],inverse = Inverse)
    
    if GazePlot ==1:
        plot_pleft_by_left_gaze_advantage(data, predictions,color1 = color_data,
                                      ax=axs[1][0])
        plot_corpleft_by_left_gaze_advantage(data, predictions, color1 = color_data,
                                         ax=axs[1][1])

    # Labels
   # for label, ax in zip(list('ABCD'), axs.ravel()):
   #     ax.text(-0.15, 1.175, label, transform=ax.transAxes,
   #             fontsize=16, fontweight='bold', va='top')

    fsize = 30
    
    for axis1 in [axs[0][0],axs[0][1],axs[1][0],axs[1][1]]:
        axis1.xaxis.label.set_fontsize(fontsize = fsize) # x label
        axis1.yaxis.label.set_fontsize(fontsize = fsize) # Y label
        axis1.tick_params(axis="x", labelsize=20)
        axis1.tick_params(axis="y", labelsize=20)
    
    patch1 = mpatches.Patch(facecolor=color_data,hatch=r'', label = label1)
    patch2 = mpatches.Patch(facecolor='#606060',hatch=r'', label = label2)

    leg = plt.legend(handles=[patch1,patch2],fontsize=25,loc = 'lower right',title = legend_label)
    leg.get_frame().set_facecolor('none')
    leg.get_frame().set_linewidth(0.0)
    leg.get_title().set_fontsize(25) #legend 'Title' fontsize

    fig.tight_layout()

    return fig, axs


def add_difficulty(df):
    """
    Compute trial difficulties and add to DataFrame.

    Maximum value - mean other values.
    In the binary case, this reduces to abs(v0 - v1).

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for item_value_i
    """

    # infer number of items
    value_cols = ([col for col in df.columns
                   if col.startswith('item_value_')])
    n_items = len(value_cols)

    values = df[value_cols].values
    values_sorted = np.sort(values, axis=1)
    difficulty = values_sorted[:, -1] - np.mean(values_sorted[:, :-1], axis=1)


    levels =  (np.max(difficulty) - np.min(difficulty))/10

    lev_label = np.arange(np.min(difficulty), np.max(difficulty) + levels,levels) 
    
    difficulty2= []
    for i in range(len(difficulty)):
         difficulty2.append( lev_label[ int(difficulty[i]//levels)] )
     
    difficulty = difficulty2
    
    df['difficulty'] = np.around(difficulty, decimals = 0)
    df['difficulty'] = difficulty

    
    return df.copy()


def plot_rt_by_difficulty(data, predictions=None, ax=None, xlims=(1.5, 8.5), xlabel_skip=2,color1 = '#4F6A9A'):
    """
    Plot SI1 Data with model predictions
    a) RT by difficulty

    Parameters
    ----------
    data: <pandas DataFrame>

    predictions: <pandas DataFrame> or <list of pandas DataFrames>

    ax: <matplotlib.axes>

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
        # Set seaborn style for the plot
        sns.set(style='white')

    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_difficulty(df)

        # Compute summary statistics
        subject_means = df.groupby(['subject', 'difficulty']).rt.mean()
        means = subject_means.groupby('difficulty').mean()[xlims[0]:xlims[1]]
        sems = subject_means.groupby('difficulty').sem()[xlims[0]:xlims[1]]

        x = np.arange(len(means))
        
        # Add labels for scatter plot of mean rt per participant
        scatter_data = subject_means.reset_index()
        x_scatter = []
        group_labels = np.sort(scatter_data.difficulty.unique())
        for ii in range(len(scatter_data.difficulty.values)):
            a = scatter_data.difficulty.values[ii]
            position_item =  x[np.where(group_labels==a)[0][0]]
            x_scatter.append(position_item) 
        ## ********    
        
        predicted = False if i == 0 else True
        
        # Colors for predicted
        c_pred = [color1,'#606060','#607681' ]
        
        if not predicted:  # plot underlying data
            ax.plot(x, means, 'o', markerfacecolor=color1, markersize = 10, fillstyle = 'full',
                    color=color1, linewidth=1)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color= color1)
            jittr = np.random.uniform(low=-max(x)/20,high=max(x)/20,size=len(scatter_data))/2
            ax.plot(x_scatter+jittr, scatter_data.rt.values, marker='o', ms=5, color=color1,alpha=0.3,linestyle="None")

        else:  # plot predictions
            ax.plot(x, means, '-o', markerfacecolor=c_pred[i],color=c_pred[i], linewidth=2.5, markersize = 10)

    #ax.set_ylim(0, 5000)
    ax.set_xlabel('|$ΔDots$|')
    ax.set_ylabel('RT (ms)')
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(np.around(means.index.values[::xlabel_skip],decimals = 1))

    despine()
    
def z_score1(data_all, part_def,z_score_var):
    z_matrix=[]
    z_matrix_aux=[]

    for i in (data_all[part_def].unique()):
        Choicedata = data_all.loc[data_all[part_def] == i]    
    
        pX_A= pd.to_numeric(Choicedata[z_score_var]) 
        pX_zA= (pX_A - np.mean(pX_A))/np.std(pX_A)

    
        z_matrix_aux= pX_zA.values
    
        for  j in range(len(z_matrix_aux)):    
            z_matrix.append(z_matrix_aux[j])
    return z_matrix
    
    
def plot_rt_by_difficulty_zSc(data, predictions=None, ax=None, xlims=(1.5, 8.5), xlabel_skip=2,color1 = '#4F6A9A'):
    """
    Plot SI1 Data with model predictions
    a) RT by difficulty

    Parameters
    ----------
    data: <pandas DataFrame>

    predictions: <pandas DataFrame> or <list of pandas DataFrames>

    ax: <matplotlib.axes>

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
        # Set seaborn style for the plot
        sns.set(style='white')

    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_difficulty(df)

        df['zrt'] = z_score1(df,'subject','rt')
        
        # Compute summary statistics
        subject_means = df.groupby(['subject', 'difficulty']).zrt.mean()
        means = subject_means.groupby('difficulty').mean()[xlims[0]:xlims[1]]
        sems = subject_means.groupby('difficulty').sem()[xlims[0]:xlims[1]]

        x = np.arange(len(means))
        
        # Add labels for scatter plot of mean rt per participant
        scatter_data = subject_means.reset_index()
        x_scatter = []
        group_labels = np.sort(scatter_data.difficulty.unique())
        for ii in range(len(scatter_data.difficulty.values)):
            a = scatter_data.difficulty.values[ii]
            position_item =  x[np.where(group_labels==a)[0][0]]
            x_scatter.append(position_item) 
        ## ********    
        
        predicted = False if i == 0 else True
        
        # Colors for predicted
        c_pred = [color1,'#606060','#607681' ]
        
        if not predicted:  # plot underlying data
            ax.plot(x, means, 'o', markerfacecolor=color1, markersize = 10, fillstyle = 'full',
                    color=color1, linewidth=1)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color= color1)
            jittr = np.random.uniform(low=-max(x)/20,high=max(x)/20,size=len(scatter_data))/2
            ax.plot(x_scatter+jittr, scatter_data.zrt.values, marker='o', ms=5, color=color1,alpha=0.3,linestyle="None")

        else:  # plot predictions
            ax.plot(x, means, '-o', markerfacecolor=c_pred[i],color=c_pred[i], linewidth=2.5, markersize = 10)

    #ax.set_ylim(2000, 3500)
    ax.set_xlabel('|$ΔDots$|')
    ax.set_ylabel('zRT (ms)')
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(np.around(means.index.values[::xlabel_skip],decimals = 1))

    despine()


def add_left_minus_mean_others(df):
    """
    Compute relative value of left item and add to DataFrame.

    Left rating – mean other ratings
    In the binary case, this reduces to v0 - v1.

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for item_value_i
    """

    # infer number of items
    value_cols = ([col for col in df.columns
                   if col.startswith('item_value_')])
    n_items = len(value_cols)

    values = df[value_cols].values
    left_minus_mean_others = values[:, 0] - np.mean(values[:, 1:], axis=1)
    
                       
   # levels =  (np.max(left_minus_mean_others) - np.min(left_minus_mean_others))/10

  #  lev_label = np.arange(np.min(left_minus_mean_others), np.max(left_minus_mean_others) + levels,levels) 
    
  #  left_minus_mean_others2= []
  #  for i in range(len(left_minus_mean_others)):
  #       left_minus_mean_others2.append( lev_label[ int(left_minus_mean_others[i]//levels)] )                   
    
    df['left_minus_mean_others'] = np.around(left_minus_mean_others,decimals= 1) 

    return df.copy()


def plot_pleft_by_left_minus_mean_others(data, predictions=None, ax=None, xlims=[-5, 5], xlabel_skip=2, xlabel_start=1, color1 = '#4F6A9A',inverse = False):
    """
    Plot SI1 Data with model predictions
    b) P(left chosen) by left rating minus mean other rating

    Parameters
    ----------
    data: <pandas DataFrame>

    predictions: <pandas DataFrame> or <list of pandas DataFrames>

    ax: <matplotlib.axes>

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
        # Set seaborn style for the plot
        sns.set(style='white')


    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    n_items = len([col for col in data.columns
                   if col.startswith('item_value_')])

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_left_minus_mean_others(df)
        df['left_chosen'] = df['choice'] == 0
        
        # Compute summary statistics
        subject_means = df.groupby(
            ['subject', 'left_minus_mean_others']).left_chosen.mean()
        means = subject_means.groupby('left_minus_mean_others').mean()[
            xlims[0]:xlims[1]]
        sems = subject_means.groupby('left_minus_mean_others').sem()[
            xlims[0]:xlims[1]]
        
        x = np.arange(len(means))
        
        # Add labels for scatter plot of mean left_minus_mean_others per participant
        scatter_data = subject_means.reset_index()

        x_scatter = []
        group_labels = np.sort(df.left_minus_mean_others.unique())
        for ii in range(len(scatter_data.left_minus_mean_others.values)):
            a = scatter_data.left_minus_mean_others.values[ii]
            position_item =  x[np.where(group_labels==a)[0][0]]
            x_scatter.append(position_item) 
        ## ********    
        
        predicted = False if i == 0 else True
        
        # Colors for predicted
        c_pred = [color1,'#606060','#607681']
        
        if not predicted:  # plot underlying data
            ax.plot(x, means, 'o', markerfacecolor=color1, markersize = 10, fillstyle = 'full',
                    color=color1, linewidth=1)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color= color1)
            jittr = np.random.uniform(low=-max(x)/20,high=max(x)/20,size=len(scatter_data))/2
            ax.plot(x_scatter+jittr, scatter_data.left_chosen.values, marker='o', ms=5, color=color1,alpha=0.3,linestyle="None")

        else:  # plot predictions
            ax.plot(x, means, '-o', markerfacecolor=c_pred[i],color=c_pred[i], linewidth=2.5, markersize = 10)

    ax.axhline(1 / n_items, linestyle='--', color='k', linewidth=1, alpha=0.2)

    ax.set_xlabel('$ΔDots$')
    ax.set_ylabel('Prob(choice = left)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x[xlabel_start::xlabel_skip])
    ax.set_xticklabels(means.index.values[xlabel_start::xlabel_skip])

    despine()


def add_left_gaze_advantage(df):
    """
    Compute gaze advantage of left item and add to DataFrame.

    Left relative gaze – mean other relative gaze
    In the binary case, this reduces to g0 - g1.

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for gaze_i
    """

    # infer number of items
    gaze_cols = ([col for col in df.columns
                  if col.startswith('gaze_')])
    n_items = len(gaze_cols)

    gaze = df[gaze_cols].values
    left_gaze_advantage_raw = gaze[:, 0] - np.mean(gaze[:, 1:], axis=1)
    df['left_gaze_advantage_raw'] = left_gaze_advantage_raw
    bins_values = []
    
    if len(np.unique(left_gaze_advantage_raw)) > 4:
        for i in (df['subject'].unique()):
            Choicedata_gaze = df.loc[df['subject'] == i]
            bins_per_subj = pd.qcut(Choicedata_gaze['left_gaze_advantage_raw'], 8,labels=False , duplicates = 'drop')
            for  j in range(len(bins_per_subj)):    
                bins_values.append(bins_per_subj.values[j])
    
        df['left_gaze_advantage'] = bins_values
    
    else: # Mod PSD/2021 to be able to properly calculate the left gaze advantage when we have fixed sampling time
        for n, i in enumerate(left_gaze_advantage_raw):
            if i == -0.5:
                left_gaze_advantage_raw[n] = 0
            if i == 0:
                left_gaze_advantage_raw[n] = 1
            if i == 0.5:
                left_gaze_advantage_raw[n] = 2
        df['left_gaze_advantage'] = left_gaze_advantage_raw
   
    df = df.drop(['left_gaze_advantage_raw'], 1)

    
    return df.copy()


def plot_pleft_by_left_gaze_advantage(data, predictions=None, ax=None, n_bins=8, xlabel_skip=2, color1 = '#4F6A9A'):
    """
    Plot SI1 Data with model predictions
    c) P(left chosen) by left gaze minus mean other gaze

    x-axis label indicate left bound of interval.

    Parameters
    ----------
    data: <pandas DataFrame>

    predictions: <pandas DataFrame> or <list of pandas DataFrames>

    ax: <matplotlib.axes>

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
        # Set seaborn style for the plot
        sns.set(style='white')
        
        
    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    n_items = len([col for col in data.columns
                   if col.startswith('item_value_')])

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        df = add_left_gaze_advantage(df)
        bins = np.linspace(0, n_bins, n_bins+1)
        

        df['left_gaze_advantage_bin'] = pd.cut(df['left_gaze_advantage'],
                                               bins=bins, include_lowest=True,
                                               labels=bins[:-1])
        
        # PSD Mod 2021: case we have only one bin therefore is not properly plotted        
        if len(df['left_gaze_advantage_bin'].unique())<4: # check if we have only one bin
            df['left_gaze_advantage_bin'] = df['left_gaze_advantage']
        
        df['left_chosen'] = df['choice'] == 0

        # Compute summary statistics
        subject_means = df.groupby(
            ['subject', 'left_gaze_advantage_bin']).left_chosen.mean()
        means = subject_means.groupby('left_gaze_advantage_bin').mean()
        sems = subject_means.groupby('left_gaze_advantage_bin').sem()

        x = np.arange(len(means))

        # Add labels for scatter plot of mean left_minus_mean_others per participant
        scatter_data = subject_means.reset_index()

        x_scatter = []
        group_labels = np.sort(df.left_gaze_advantage_bin.unique())
        for ii in range(len(scatter_data.left_gaze_advantage_bin.values)):
            a = scatter_data.left_gaze_advantage_bin.values[ii]
            position_item =  x[np.where(group_labels==a)[0][0]]
            x_scatter.append(position_item) 
        ## ********            

        predicted = False if i == 0 else True
        
        # Colors for predicted
        c_pred = [color1,'#606060','#607681']
        
        print()
        
        if not predicted:  # plot underlying data
            ax.plot(x, means, 'o', markerfacecolor=color1, markersize = 10, fillstyle = 'full',
                    color=color1, linewidth=1)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color= color1)
            jittr = np.random.uniform(low=-max(x)/20,high=max(x)/20,size=len(scatter_data))/2
            ax.plot(x_scatter+jittr, scatter_data.left_chosen.values, marker='o', ms=5, color=color1,alpha=0.3,linestyle="None", linewidth=5)

        else:  # plot predictions

            ax.plot(x, means, '-o', markerfacecolor=c_pred[i],color=c_pred[i], linewidth=2.5, markersize = 10)

    ax.set_xlabel('$Δ Gaze_{Bins}$')
    ax.set_ylabel('Prob(choice = left)')
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(means.index.values[::xlabel_skip])

    despine()


def add_left_relative_value(df):
    """
    Compute relative value of left item.

    Left item value – mean other item values
    In the binary case, this reduces to v0 - v1.

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for gaze_i
    """

    # infer number of items
    # relative value left
    value_cols = ([col for col in df.columns
                   if col.startswith('item_value_')])
    n_items = len(value_cols)
    values = df[value_cols].values
    relative_value_left = values[:, 0] - np.mean(values[:, 1:])
    df['left_relative_value'] = relative_value_left

    return df.copy()


def add_corrected_choice_left(df):
    """
    Compute corrected choice left

    Corrected choice ~ (choice==left) - p(choice==left | left relative item value)

    Parameters
    ----------
    df :      <pandas DataFrame>
              Trial wise DataFrame containing columns for gaze_i
    """

    # recode choice
    df['left_chosen'] = df['choice'].values == 0

    # left relative value
    df = add_left_relative_value(df)

    # compute p(choice==left|left relative value)
    subject_value_psychometric = df.groupby(
        ['subject', 'left_relative_value']).left_chosen.mean()
    # place in dataframe
    for s, subject in enumerate(df['subject'].unique()):
        subject_df = df[df['subject'] == subject].copy()
        df.loc[df['subject'] == subject, 'p_choice_left_given_value'] = subject_value_psychometric[
            subject][subject_df['left_relative_value'].values].values

    # compute corrected choice left
    df['corrected_choice_left'] = df['left_chosen'] - df['p_choice_left_given_value']

    return df.copy()


def plot_corpleft_by_left_gaze_advantage(data, predictions=None, ax=None, n_bins=8, xlabel_skip=2, color1 = '#4F6A9A'):
    """
    Plot SI1 Data with model predictions
    c) Corrected P(choice==left) by left gaze minus mean other gaze
    Corrected P(choice==left) ~ P(choice==left | left final gaze adv.) - P(choice==left | left relative value)

    Parameters
    ----------
    data: <pandas DataFrame>

    predictions: <pandas DataFrame> or <list of pandas DataFrames>

    ax: <matplotlib.axes>

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
        # Set seaborn style for the plot
        sns.set(style='white')
        
        
    if predictions is None:
        dataframes = [data]
    elif isinstance(predictions, list):
        dataframes = [data] + predictions
    else:
        dataframes = [data] + [predictions]

    n_items = len([col for col in data.columns
                   if col.startswith('item_value_')])

    for i, dataframe in enumerate(dataframes):

        df = dataframe.copy()

        # Compute relevant variables
        # recode choice
        df['left_chosen'] = df['choice'].values == 0
        # left final gaze advantage
        df = add_left_gaze_advantage(df)
        gaze_bins = np.linspace(0, n_bins, n_bins+1)
        df['left_gaze_advantage_bin'] = pd.cut(df['left_gaze_advantage'],
                                               bins=gaze_bins, include_lowest=True,
                                              labels=gaze_bins[:-1])
       
        # PSD Mod 2021: case we have only one bin therefore is not properly plotted        
        if len(df['left_gaze_advantage_bin'].unique())<4: # check if we have only one bin
            df['left_gaze_advantage_bin'] = df['left_gaze_advantage']

        df['left_chosen'] = df['choice'] == 0
        # corrected choice
        df = add_corrected_choice_left(df)

        # Compute summary statistics
        subject_means = df.groupby(
            ['subject', 'left_gaze_advantage_bin']).corrected_choice_left.mean()
        means = subject_means.groupby('left_gaze_advantage_bin').mean()
        sems = subject_means.groupby('left_gaze_advantage_bin').sem()
        x = np.arange(len(means))

        # Add labels for scatter plot of mean left_minus_mean_others per participant
        scatter_data = subject_means.reset_index()

        x_scatter = []
        group_labels = np.sort(df.left_gaze_advantage_bin.unique())
        for ii in range(len(scatter_data.left_gaze_advantage_bin.values)):
            a = scatter_data.left_gaze_advantage_bin.values[ii]
            position_item =  x[np.where(group_labels==a)[0][0]]
            x_scatter.append(position_item) 
        ## ********            
        
        predicted = False if i == 0 else True
        
        # Colors for predicted
        c_pred = [color1,'#606060','#607681' ]

        if not predicted:  # plot underlying data
            ax.plot(x, means, 'o', markerfacecolor=color1, markersize = 10, fillstyle = 'full',
                    color=color1, linewidth=1)
            ax.vlines(x, means - sems, means + sems,
                      linewidth=1, color= color1)
            jittr = np.random.uniform(low=-max(x)/20,high=max(x)/20,size=len(scatter_data))/2
            ax.plot(x_scatter+jittr, scatter_data.corrected_choice_left.values, marker='o', ms=5, color=color1,alpha=0.3,linestyle="None")

        else:  # plot predictions
            ax.plot(x, means, '-o', markerfacecolor=c_pred[i],color=c_pred[i], linewidth=2.5, markersize = 10)

    ax.set_xlabel('$Δ Gaze_{Bins}$')
    ax.set_ylabel('Corrected Prob(choice = left)')
    ax.set_xticks(x[::xlabel_skip])
    ax.set_xticklabels(means.index.values[::xlabel_skip])
    ax.set_ylim(-.4, .4)

    despine()