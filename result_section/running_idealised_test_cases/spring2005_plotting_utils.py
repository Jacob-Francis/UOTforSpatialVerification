import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import datetime, timedelta
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import spearmanr


def data_pull(key, rho=1):
    """return: wrf2caps, wrf4ncar, wrf4ncep
    All are TV thn KL divergence.
    Parameters
    ----------
    key : 'Se', 'p1', 'p2', 'p3', 'p4', 'd1', 'd2',
       'd3', 'primal', 'dual', 'loop_timing', 'forward_mean_mag_p',
       'forward_median_mag_p', 'forward_mean_dir_p', 'forward_median_dir_p',
       'forward_mean_mag_se', 'forward_median_mag_se', 'forward_mean_dir_se',
       'forward_median_dir_se', 'inverse_mean_mag_p', 'inverse_median_mag_p',
       'inverse_mean_dir_p', 'inverse_median_dir_p', 'inverse_mean_mag_se',
       'inverse_median_mag_se', 'inverse_mean_dir_se', 'inverse_median_dir_se',
       'mass_x', 'mass_y'
    """
    
    wrf2caps = np.zeros((2, 9))
    wrf4ncar = np.zeros((2, 9))
    wrf4ncep = np.zeros((2, 9))

    x_titles = []

    if key in ['Se']:
        factor = 601**2
    else:
        factor = 601

    _files = os.listdir('/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store')
    files = sorted(_files)
        
    i = 0

    for file in files:
        if f'rho_{rho}_' in file:
            x_titles.append(file.split('_')[-1].split('.')[0])
            data=pd.read_csv('/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store/'+file)
            # f = open('/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store/' + file, 'rb')
            # data = pickle.load(f)
            # f.close()
            for keys in set(data.case_key):
                if keys=='st2/ST2ml_2005042600.g240.txtst2/ST2ml_2005042600.g240.txt':
                    pass
                else:
                    if 'wrf2caps' in keys:
                        wrf2caps[:, i] = data.loc[data.case_key==keys][key].values*factor
                    if 'wrf4ncar' in keys:
                        wrf4ncar[:, i] = data.loc[data.case_key==keys][key].values*factor
                    if 'wrf4ncep' in keys:
                        wrf4ncep[:, i] = data.loc[data.case_key==keys][key].values*factor
            
            i+=1
    
    # check dates in correct order
    assert((x_titles == ['2005042500',
                '2005051200',
                '2005051300',
                '2005051700',
                '2005051800',
                '2005052400',
                '2005053100',
                '2005060200',
                '2005060300']))
    return wrf2caps, wrf4ncar, wrf4ncep

def plot_transport_vs_balance():
    cases=np.unique(data.case_key)
    cases
    plt.figure(figsize=(8, 6), dpi=200)
    texts = []
    # Initialize storage dictionaries
    transport_dict = {k: [] for k in cases}
    balance_dict = {k: [] for k in cases}

    rho_values = [1, 0.125, 0.0153125]
    markers = ['^', 's', 'o']  # Triangle, Square, Circle
    colours = ['k', 'k', 'k', 'k','k', 'k', 'r', 'r']  # Black color for all


    for case in cases:

        for rho in [1,0.125,0.0153125]:
            data=pd.read_csv('/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store/'+file)

            wrf2caps, wrf4ncar, wrf4ncep = pd.read_csv(f'/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/idealised_fake_cases/data_store/debiased_cases_0.001_rho_{rho}_scaling_prturbed_fake_cases.csv')


            # Extract transport and balance values
            transport_value = np.abs(data.loc[(data.case_key == case) & (data.aprox_type == 'tv')].p1.values)
            balance_value = (data.loc[(data.case_key == case) & (data.aprox_type == 'tv')].p3.values + 
                            data.loc[(data.case_key == case) & (data.aprox_type == 'tv')].p4.values)/rho

            transport_dict[case].append(transport_value)
            balance_dict[case].append(balance_value)

    for l, case in enumerate(cases):
        plt.loglog(balance_dict[case], transport_dict[case], 'k:')
        txt = plt.annotate(f'{case[7:]}', 
                        (balance_dict[case][1], transport_dict[case][1]),fontsize=10)
        texts.append(txt)  # Store text objects for later adjustment

        for k, (i, j) in enumerate(zip(balance_dict[case], transport_dict[case])):
            plt.scatter(i, j, marker=markers[k], edgecolors=colours[l], facecolors='none', s=80, label=f'ρ = {rho_values[k]}' if case == cases[0] else "")


    adjust_text(texts, force_explode=2)
    # Labels and title
    plt.xlabel('Marginal Mass Imbalance Term')
    plt.legend(title='ρ')
    plt.ylabel('Transport Term')
    plt.title(r'Transport vs. Mass Balance Term Across Cases (TV, loglog scale)')
    # plt.legend(title='Case')
    # Define custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='k', linestyle='None', markersize=8, label=r'$\rho = 1$ '),
        plt.Line2D([0], [0], marker='s', color='k', linestyle='None', markersize=8, label=r'$\rho = 0.125$ '),
        plt.Line2D([0], [0], marker='o', color='k', linestyle='None', markersize=8, label=r'$\rho = 0.0153125$ '),
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.grid()
    
def plot_lineplots_of_scores():
    fig = plt.figure(figsize=(12,5), dpi=200)
    axkl = fig.add_subplot(211)
    axrkl = fig.add_subplot(212, sharex=axkl)

    colors = ['#1b9e77', '#d95f02', '#7570b3']

    wrf2caps = np.zeros((4, 9))
    wrf4ncar = np.zeros((4, 9))
    wrf4ncep = np.zeros((4, 9))

    x_titles = []

    _files = os.listdir('/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store')
    files = sorted(_files)
        
    i = 0

    for file in files:
        if 'rho_1_' in file:
            x_titles.append((datetime.strptime(file.split('_')[-1].split('.')[0], '%Y%m%d%H')+timedelta(days=1)).date())
            data=pd.read_csv('/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store/'+file)
            # f = open('/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store/' + file, 'rb')
            # data = pickle.load(f)
            # f.close()
            for keys in set(data.case_key):
                if keys=='st2/ST2ml_2005042600.g240.txtst2/ST2ml_2005042600.g240.txt':
                    pass
                else:
                    if 'wrf2caps' in keys:
                        wrf2caps[:2, i] = data.loc[data.case_key==keys].Se.values*601**2
                        wrf2caps[2:, i] = data.loc[data.case_key==keys].forward_median_mag_se.values*601
                    if 'wrf4ncar' in keys:
                        wrf4ncar[:2, i] = data.loc[data.case_key==keys].Se.values*601**2
                        wrf4ncar[2:, i] = data.loc[data.case_key==keys].forward_median_mag_se.values*601

                    if 'wrf4ncep' in keys:
                        wrf4ncep[:2, i] = data.loc[data.case_key==keys].Se.values*601**2
                        wrf4ncep[2:, i] = data.loc[data.case_key==keys].forward_median_mag_se.values*601
            
            i+=1

    # Plot the data
    axkl.plot(wrf2caps[0, :], '.:', color=colors[0], label='wrf2caps')
    axrkl.plot(wrf2caps[1, :], '.:', color=colors[0])

    axkl.plot(wrf4ncar[0, :], 'v--', color=colors[1], label='wrf4ncar')
    axrkl.plot(wrf4ncar[1, :], 'v--', color=colors[1])


    axkl.plot(wrf4ncep[0, :], 'x-', color=colors[2], label='wrf4ncep')
    axrkl.plot(wrf4ncep[1, :], 'x-', color=colors[2])

    axkl.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    axrkl.set_xticks(np.arange(len(x_titles)))
    axrkl.set_xticklabels(x_titles)

    axkl.legend()

    axkl.set(ylabel=r'$S_{\epsilon}^{TV}$')
    axrkl.set(ylabel=r'$S_{\epsilon}^{KL}$')
    axkl.set_yscale('log')
    axrkl.set_yscale('log')

    axrkl.grid()
    axkl.grid()

    plt.savefig('real_model_costsonly.pdf')


def plot_other_lines_plots():
    wrf2caps = np.zeros((4, 9))
    wrf4ncar = np.zeros((4, 9))
    wrf4ncep = np.zeros((4, 9))

    x_titles = []
    files = sorted(os.listdir('/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store'))
        
    i = 0
    for file in files:
        if 'rho_1_' in file:
            x_titles.append((datetime.strptime(file.split('_')[-1].split('.')[0], '%Y%m%d%H')+timedelta(days=1)).date())
            data = pd.read_csv('/home/jacob/PhD_jobs/nvidia_cluster/icp_geometric_cases/spring2005unit_grid_cas/data_store/' + file)
            for keys in set(data.case_key):
                if keys == 'st2/ST2ml_2005042600.g240.txtst2/ST2ml_2005042600.g240.txt':
                    continue
                if 'wrf2caps' in keys:
                    wrf2caps[:2, i] = data.loc[data.case_key==keys].forward_median_mag_se.values * 601
                    wrf2caps[2:, i] = data.loc[data.case_key==keys].inverse_median_mag_se.values * 601
                if 'wrf4ncar' in keys:
                    wrf4ncar[:2, i] = data.loc[data.case_key==keys].forward_median_mag_se.values * 601
                    wrf4ncar[2:, i] = data.loc[data.case_key==keys].inverse_median_mag_se.values * 601
                if 'wrf4ncep' in keys:
                    wrf4ncep[:2, i] = data.loc[data.case_key==keys].forward_median_mag_se.values * 601
                    wrf4ncep[2:, i] = data.loc[data.case_key==keys].inverse_median_mag_se.values * 601
            i += 1

    # Colour-blind & greyscale safe colours (Okabe-Ito palette)
    colours = ['#E69F00', '#56B4E9', '#009E73']  # orange, blue, green

    # Marker styles
    forward_marker = 'o'
    inverse_marker = 's'

    # Alpha values
    forward_alpha = 0.9
    inverse_alpha = 0.6

    # Plot setup
    fig = plt.figure(figsize=(12, 5), dpi=200)
    axkl = fig.add_subplot(211)
    axrkl = fig.add_subplot(212, sharex=axkl)

    # Forward lines
    axkl.plot(wrf2caps[0, :], marker=forward_marker, linestyle='-', color=colours[0], alpha=forward_alpha)
    axkl.plot(wrf4ncar[0, :], marker=forward_marker, linestyle='--', color=colours[1], alpha=forward_alpha)
    axkl.plot(wrf4ncep[0, :], marker=forward_marker, linestyle=':', color=colours[2], alpha=forward_alpha)

    axrkl.plot(wrf2caps[1, :], marker=forward_marker, linestyle='-', color=colours[0], alpha=forward_alpha)
    axrkl.plot(wrf4ncar[1, :], marker=forward_marker, linestyle='--', color=colours[1], alpha=forward_alpha)
    axrkl.plot(wrf4ncep[1, :], marker=forward_marker, linestyle=':', color=colours[2], alpha=forward_alpha)

    # Inverse lines
    axkl.plot(wrf2caps[2, :], marker=inverse_marker, linestyle='-', color=colours[0], alpha=inverse_alpha)
    axkl.plot(wrf4ncar[2, :], marker=inverse_marker, linestyle='--', color=colours[1], alpha=inverse_alpha)
    axkl.plot(wrf4ncep[2, :], marker=inverse_marker, linestyle=':', color=colours[2], alpha=inverse_alpha)

    axrkl.plot(wrf2caps[3, :], marker=inverse_marker, linestyle='-', color=colours[0], alpha=inverse_alpha)
    axrkl.plot(wrf4ncar[3, :], marker=inverse_marker, linestyle='--', color=colours[1], alpha=inverse_alpha)
    axrkl.plot(wrf4ncep[3, :], marker=inverse_marker, linestyle=':', color=colours[2], alpha=inverse_alpha)

    axkl.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axrkl.set_xticks(np.arange(len(x_titles)))
    axrkl.set_xticklabels(x_titles, rotation=45, ha='right')

    axkl.set(ylabel=r'ATM (TV)')
    axrkl.set(ylabel=r'ATM (KL)')
    # axkl.set_yscale('log')
    # axrkl.set_yscale('log')
    axrkl.grid()
    axkl.grid()

    # Custom legend/key
    legend_elements = [
        Line2D([0], [0], color=colours[0], linestyle='-', lw=2, label='wrf2caps'),
        Line2D([0], [0], color=colours[1], linestyle='--', lw=2, label='wrf4ncar'),
        Line2D([0], [0], color=colours[2], linestyle=':', lw=2, label='wrf4ncep'),
        Line2D([0], [0], marker=forward_marker, color='black', linestyle='none', markersize=9,
               markerfacecolor='black', alpha=forward_alpha, label='Forward'),
        Line2D([0], [0], marker=inverse_marker, color='black', linestyle='none', markersize=9,
               markerfacecolor='black', alpha=inverse_alpha, label='Inverse')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    plt.savefig('real_model_extracosts_cb_safe.pdf', bbox_inches='tight')


# Function to plot ranking trends
def plot_ranking_trends(ax, model_name, exp_rank, kl_rank, tv_rank, color):
    ax.bar(index - bar_width, exp_rank, bar_width, label='Expert Score', color=color[0], alpha=0.8)
    ax.bar(index, kl_rank, bar_width, label='KL Score', color=color[1], alpha=0.8)
    ax.bar(index + bar_width, tv_rank, bar_width, label='TV Score', color=color[2], alpha=0.8)

    # Connect ranking trends with lines
    # ax.plot(index, exp_rank, 'o-', color=color[0], alpha=0.8)
    # ax.plot(index, kl_rank, 's--', color=color[1], alpha=0.8)
    # ax.plot(index, tv_rank, 'x-', color=color[2], alpha=0.8)

    ax.set_ylabel('Rank (Lower is Better)')
    ax.set_title(f'Model: {model_name}')
    # ax.invert_yaxis()  # Lower rank (1) is at the top
    ax.legend()

# Define color palette (Colorblind-Safe)
colors = [
    ('#d9f0a3', '#78c679', '#238443'),  # Light to dark green (wrf2caps)
   ('#d9f0a3', '#78c679', '#238443'),  # Light to dark orange (wrf4ncar)
    ('#d9f0a3', '#78c679', '#238443') # Light to dark blue (wrf4ncep)
]

def grab_expert_data():
    # Create the DataFrame
        # assert((x_titles == ['2005042500',
        #         '2005051200',
        #         '2005051300',
        #         '2005051700',
        #         '2005051800',
        #         '2005052400',
        #         '2005053100',
        #         '2005060200',
        #         '2005060300']))
    data = {
        "Date": ["26 Apr", "26 Apr", "26 Apr", "13 May", "13 May", "13 May",
                "14 May", "14 May", "14 May", "18 May", "18 May", "18 May",
                "19 May", "19 May", "19 May", "25 May", "25 May", "25 May",
                "1 Jun", "1 Jun", "1 Jun", "3 Jun", "3 Jun", "3 Jun",
                "4 Jun", "4 Jun", "4 Jun"],
        "Model": ["wrf2caps", "wrf4ncar", "wrf4ncep", "wrf2caps", "wrf4ncar", "wrf4ncep",
                "wrf2caps", "wrf4ncar", "wrf4ncep", "wrf2caps", "wrf4ncar", "wrf4ncep",
                "wrf2caps", "wrf4ncar", "wrf4ncep", "wrf2caps", "wrf4ncar", "wrf4ncep",
                "wrf2caps", "wrf4ncar", "wrf4ncep", "wrf2caps", "wrf4ncar", "wrf4ncep",
                "wrf2caps", "wrf4ncar", "wrf4ncep"],
        "Expert Score": [3.19, 3.22, 3.40, 3.62, 3.61, 2.35,
                        2.62, 2.78, 2.28, 3.19, 3.28, 3.32,
                        2.17, 2.17, 2.93, 2.80, 2.58, 2.72,
                        3.46, 3.48, 3.03, 2.89, 2.94, 3.19,
                        2.49, 3.02, 2.10],
        "DAS": [0.81, 0.83, 0.81, 1.18, 1.12, 1.38, 0.99,
                1.08, 1.3, 1.09, 1.05, 1.05, 0.97, 1.08,
                0.83, 1.1, 1.22, 1.11, 1.28, 1.24, 1.22,
                    0.94, 0.94, 0.86, 1.14, 0.91, 1.26]
    }

    return pd.DataFrame(data)

def plot_ranking_bars():
    df = grab_expert_data()
    wrf2caps_exprt = df.loc[df.Model == 'wrf2caps']['Expert Score'].values
    wrf4ncar_exprt =  df.loc[df.Model == 'wrf4ncar']['Expert Score'].values
    wrf4ncep_exprt =  df.loc[df.Model == 'wrf4ncep']['Expert Score'].values

    fig  = plt.figure(figsize=(12, 15))
    bar_width = 0.25
    colors = [
        ('#d9f0a3', '#78c679', '#238443'),  # Light to dark green (wrf2caps)
    ('#d9f0a3', '#78c679', '#238443'),  # Light to dark orange (wrf4ncar)
        ('#d9f0a3', '#78c679', '#238443') # Light to dark blue (wrf4ncep)
    ]
    # Expert rank :
    ranking_exprt = 3 - np.argsort([wrf2caps_exprt, wrf4ncar_exprt, wrf4ncep_exprt],  axis=0) 

    ax = fig.add_subplot(13,1,1)
    ax.set_title('Expert Ranking', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking_exprt[0, index], bar_width, label='wrf2caps', color=colors[0][0])
        ax.bar(index, ranking_exprt[1, index], bar_width, label='wrf4ncar', color=colors[0][1])
        ax.bar(index + bar_width, ranking_exprt[2, index], bar_width, label='wrf4ncep', color=colors[0][2])


    wrf2caps, wrf4ncar, wrf4ncep = data_pull('Se', rho=1)

    # tv rank :
    ranking = np.argsort([wrf2caps[0], wrf4ncar[0], wrf4ncep[0]],  axis=0) + 1

    ax = fig.add_subplot(13,1,2,sharex=ax)
    ax.set_title(f'TV (Se) : {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    # kl rank :
    ranking = np.argsort([wrf2caps[1], wrf4ncar[1], wrf4ncep[1]],  axis=0) + 1

    ax = fig.add_subplot(13,1,3,sharex=ax)
    ax.set_title(f'KL (Se) : {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    wrf2caps, wrf4ncar, wrf4ncep = data_pull('p1', rho=1)

    # tv rank :
    ranking = np.argsort([wrf2caps[0], wrf4ncar[0], wrf4ncep[0]],  axis=0) + 1

    ax = fig.add_subplot(13,1,4,sharex=ax)
    ax.set_title(f'TV (p1) : {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    # kl rank :
    ranking = np.argsort([wrf2caps[1], wrf4ncar[1], wrf4ncep[1]],  axis=0) + 1

    ax = fig.add_subplot(13,1,5,sharex=ax)
    ax.set_title(f'KL (p1) : {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])


    wrf2caps, wrf4ncar, wrf4ncep = data_pull('forward_median_mag_se', rho=1)

    # tv rank :
    ranking = np.argsort([wrf2caps[0], wrf4ncar[0], wrf4ncep[0]],  axis=0) + 1

    ax = fig.add_subplot(13,1,6,sharex=ax)
    ax.set_title(f'TV (ATM, median)'+f': \n        {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    # kl rank :
    ranking = np.argsort([wrf2caps[1], wrf4ncar[1], wrf4ncep[1]],  axis=0) + 1

    ax = fig.add_subplot(13,1,7,sharex=ax)
    ax.set_title(f'KL (ATM, median)'+f': \n        {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    wrf2caps, wrf4ncar, wrf4ncep = data_pull('forward_mean_mag_se', rho=1)

    # tv rank :
    ranking = np.argsort([wrf2caps[0], wrf4ncar[0], wrf4ncep[0]],  axis=0) + 1

    ax = fig.add_subplot(13,1,8,sharex=ax)
    ax.set_title(f'TV (ATM, mean)'+f': \n        {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    # kl rank :
    ranking = np.argsort([wrf2caps[1], wrf4ncar[1], wrf4ncep[1]],  axis=0) + 1

    ax = fig.add_subplot(13,1,9,sharex=ax)
    ax.set_title(f'KL (ATM, mean)'+f': \n        {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    wrf2caps, wrf4ncar, wrf4ncep = data_pull('inverse_median_mag_se', rho=1)

    # tv rank :
    ranking = np.argsort([wrf2caps[0], wrf4ncar[0], wrf4ncep[0]],  axis=0) + 1

    ax = fig.add_subplot(13,1,10,sharex=ax)
    ax.set_title(r'TV (ATM$^{-1}$, median)'+f': \n        {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    # kl rank :
    ranking = np.argsort([wrf2caps[1], wrf4ncar[1], wrf4ncep[1]],  axis=0) + 1

    ax = fig.add_subplot(13,1,11,sharex=ax)
    ax.set_title(r'KL (ATM$^{-1}$, median)'+f': \n        {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    wrf2caps, wrf4ncar, wrf4ncep = data_pull('inverse_mean_mag_se', rho=1)

    # tv rank :
    ranking = np.argsort([wrf2caps[0], wrf4ncar[0], wrf4ncep[0]],  axis=0) + 1

    ax = fig.add_subplot(13,1,12,sharex=ax)
    ax.set_title(r'TV (ATM$^{-1}$, mean)'+f': \n        {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    # kl rank :
    ranking = np.argsort([wrf2caps[1], wrf4ncar[1], wrf4ncep[1]],  axis=0) + 1

    ax = fig.add_subplot(13,1,13,sharex=ax)
    ax.set_title(r'KL (ATM$^{-1}$, mean)'+f': \n        {sum(sum(ranking==ranking_exprt))}/27', loc='left', x=-0.2,y=0.35)
    for index in range(9):
        ax.bar(index - bar_width, ranking[0, index], bar_width, label='wrf2caps', color=colors[0][0], hatch=['/' if ranking[0, index] == ranking_exprt[0, index] else None])
        ax.bar(index, ranking[1, index], bar_width, label='wrf4ncar', color=colors[0][1], hatch=['/' if ranking[1, index] == ranking_exprt[1, index] else None])
        ax.bar(index + bar_width, ranking[2, index], bar_width, label='wrf4ncep', color=colors[0][2], hatch=['/' if ranking[2, index] == ranking_exprt[2, index] else None])

    dates = ['2005042500',
                    '2005051200',
                    '2005051300',
                    '2005051700',
                    '2005051800',
                    '2005052400',
                    '2005053100',
                    '2005060200',
                    '2005060300']
    model_labels = ['wrf2caps', 'wrf4ncar', 'wrf4ncep']

    import itertools

    x_positions = [(k - bar_width, k, k + bar_width) for k in np.arange(9)]

    # Flatten the list
    flat_x_positions = list(itertools.chain.from_iterable(x_positions))

    # OR using numpy
    flat_x_positions = np.ravel(x_positions).tolist()

    for ax in fig.get_axes():
        if ax != fig.get_axes()[-1]:  # Hide x-axis ticks except on the last subplot
            ax.label_outer()
        else:
            ax.set_xticks(flat_x_positions)  # Centered around the first model
            ax.set_xticklabels(model_labels * 9, rotation=90)  # Model names under each bar
            
            # Set secondary x-axis for dates
            ax_secondary = ax.secondary_xaxis('bottom')
            ax_secondary.set_xticks(np.arange(9))
            ax_secondary.set_xticklabels(dates, rotation=45, ha='right')  # Rotated for clarity
            # ax_secondary.set_xlabel("Dates")
            ax_secondary.tick_params(axis='x', pad=50)

    plt.suptitle(r'$\rho=1$')
    plt.tight_layout()
    plt.savefig('ranking_trends_rho1.pdf', dpi=300, bbox_inches='tight')


def plot_ranking_spearmans_overdays():

    df = grab_expert_data()
    # Expert scores
    wrf2caps_exprt = df.loc[df.Model == 'wrf2caps']['Expert Score'].values
    wrf4ncar_exprt = df.loc[df.Model == 'wrf4ncar']['Expert Score'].values
    wrf4ncep_exprt = df.loc[df.Model == 'wrf4ncep']['Expert Score'].values

    models = ['wrf2caps', 'wrf4ncar', 'wrf4ncep']
    experts = [-wrf2caps_exprt, -wrf4ncar_exprt, -wrf4ncep_exprt]
    experts = - np.argsort([wrf2caps_exprt, wrf4ncar_exprt, wrf4ncep_exprt],  axis=0) 

    metrics = ['Se', 'p1', 'forward_median_mag_se', 'forward_mean_mag_se', 'inverse_median_mag_se', 'inverse_mean_mag_se']
    titles = [
        "TV (Se)", "KL (Se)", 
        "TV (p1)", "KL (p1)", 
        "TV (ATM, median)", "KL (ATM, median)", 
        "TV (ATM, mean)", "KL (ATM, mean)",
        "TV (ATM$^{-1}$, median)", "KL (ATM$^{-1}$, median)", 
        "TV (ATM$^{-1}$, mean)", "KL (ATM$^{-1}$, mean)",
    ]
    times = ['2005042500', '2005051200', '2005051300', '2005051700',
            '2005051800', '2005052400', '2005053100', '2005060200', '2005060300']

    results_tv = []
    results_kl = []

    i = 0
    for metric in metrics:
        wrf_vals = data_pull(metric, rho=1)  # returns list of 3 arrays (one per model)

        for time_idx, time in enumerate(times):
            # For this time, get each model’s value for the metric
            model_vals_tv = [vals[0][time_idx] for vals in wrf_vals]
            model_vals_kl = [vals[1][time_idx] for vals in wrf_vals]

            # Expert scores for this time
            expert_scores = [expert[time_idx] for expert in experts] 

            # Spearman rank between expert ranking and metric ranking across models
            rho_tv, pval_tv = spearmanr(expert_scores, model_vals_tv)
            rho_kl, pval_kl = spearmanr(expert_scores, model_vals_kl)

            results_tv.append({'Metric': titles[i], 'Time': time, 'Spearman ρ': rho_tv, 'p-value': pval_tv})
            results_kl.append({'Metric': titles[i+1], 'Time': time, 'Spearman ρ': rho_kl, 'p-value': pval_kl})

        i += 2

    # Convert to DataFrames
    corr_df_tv = pd.DataFrame(results_tv)
    corr_df_kl = pd.DataFrame(results_kl)

    # Updated heatmap plotting function: Metrics as rows, Times as columns
    def plot_corr_heatmap(corr_df, title, save_title):
        plt.figure(figsize=(9, 6), dpi=200)
        heatmap_data = corr_df.pivot(index='Metric', columns='Time', values='Spearman ρ')
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                    cbar_kws={'label': 'Spearman ρ'}, linewidths=0.5, linecolor='black', fmt=".2f")
        plt.title(title)
        plt.ylabel("Transport Metric")
        plt.xlabel("Time")
        plt.tight_layout()
        plt.savefig(save_title)
        # plt.show()

    # Plot both
    plot_corr_heatmap(corr_df_tv, "Spearman Rank Correlation (TV)", "TV_Correlation_Heatmap_3.pdf")
    plot_corr_heatmap(corr_df_kl, "Spearman Rank Correlation (KL)", "KL_Correlation_Heatmap_3.pdf")


def plot_ranking_spearmans_acrossmodels():
    df = grab_expert_data()

    # Expert scores
    wrf2caps_exprt = df.loc[df.Model == 'wrf2caps']['Expert Score'].values
    wrf4ncar_exprt = df.loc[df.Model == 'wrf4ncar']['Expert Score'].values
    wrf4ncep_exprt = df.loc[df.Model == 'wrf4ncep']['Expert Score'].values

    models = ['wrf2caps', 'wrf4ncar', 'wrf4ncep']
    experts = [-wrf2caps_exprt, -wrf4ncar_exprt, -wrf4ncep_exprt]
    # experts = 9 - np.argsort([wrf2caps_exprt, wrf4ncar_exprt, wrf4ncep_exprt],  axis=0) 

    metrics = ['Se', 'p1', 'forward_median_mag_se', 'forward_mean_mag_se', 'inverse_median_mag_se', 'inverse_mean_mag_se']
    titles = [
        "TV (Se)", "KL (Se)", 
        "TV (p1)", "KL (p1)", 
        "TV (ATM, median)", "KL (ATM, median)", 
        "TV (ATM, mean)", "KL (ATM, mean)",
        "TV (ATM$^{-1}$, median)", "KL (ATM$^{-1}$, median)", 
        "TV (ATM$^{-1}$, mean)", "KL (ATM$^{-1}$, mean)",
    ]
    times = ['2005042500',
                    '2005051200',
                    '2005051300',
                    '2005051700',
                    '2005051800',
                    '2005052400',
                    '2005053100',
                    '2005060200',
                    '2005060300']

    results_tv = []
    results_kl = []

    i = 0
    for metric in metrics:
        wrf_vals = data_pull(metric, rho=1)

        for model, expert, vals in zip(models, experts, wrf_vals):
            # TV rank correlation (index 0)
            rho_tv, pval_tv = spearmanr(expert, vals[0])
            results_tv.append({'Metric': titles[i], 'Model': model, 'Spearman ρ': rho_tv, 'p-value': pval_tv})

            # KL rank correlation (index 1)
            rho_kl, pval_kl = spearmanr(expert, vals[1])
            results_kl.append({'Metric': titles[i+1], 'Model': model, 'Spearman ρ': rho_kl, 'p-value': pval_kl})

        i += 2

    # Convert to DataFrames
    corr_df_tv = pd.DataFrame(results_tv)
    corr_df_kl = pd.DataFrame(results_kl)

    # Updated heatmap plotting function with swapped axes
    def plot_corr_heatmap(corr_df, title, save_title):
        plt.figure(figsize=(6, 6), dpi=200)
        heatmap_data = corr_df.pivot(index='Metric', columns='Model', values='Spearman ρ')  # <-- swapped 'Model' and 'Metric' here
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                    cbar_kws={'label': 'Spearman ρ'}, linewidths=0.5, linecolor='black', fmt=".2f")
        plt.title(title)
        plt.ylabel("")  # <-- adjusted label
        plt.xlabel("Model")      # <-- adjusted label
        plt.tight_layout()
        plt.savefig(save_title)

    # Plot both
    plot_corr_heatmap(corr_df_tv, "Spearman Rank Correlation (TV)", save_title="acrossmodel_TV_spearman_heatmap_2.pdf")
    plot_corr_heatmap(corr_df_kl, "Spearman Rank Correlation (KL)", save_title="acrossmodel_Kl_spearman_heatmap_2.pdf")

from scipy.stats import kendalltau  # <-- import kendalltau

def plot_ranking_kendalls_acrossmodels():
    df = grab_expert_data()

    # Expert scores
    wrf2caps_exprt = df.loc[df.Model == 'wrf2caps']['Expert Score'].values
    wrf4ncar_exprt = df.loc[df.Model == 'wrf4ncar']['Expert Score'].values
    wrf4ncep_exprt = df.loc[df.Model == 'wrf4ncep']['Expert Score'].values

    models = ['wrf2caps', 'wrf4ncar', 'wrf4ncep']
    experts = [-wrf2caps_exprt, -wrf4ncar_exprt, -wrf4ncep_exprt]

    metrics = ['Se', 'p1', 'forward_median_mag_se', 'forward_mean_mag_se', 'inverse_median_mag_se', 'inverse_mean_mag_se']
    titles = [
        "TV (Se)", "KL (Se)", 
        "TV (p1)", "KL (p1)", 
        "TV (ATM, median)", "KL (ATM, median)", 
        "TV (ATM, mean)", "KL (ATM, mean)",
        "TV (ATM$^{-1}$, median)", "KL (ATM$^{-1}$, median)", 
        "TV (ATM$^{-1}$, mean)", "KL (ATM$^{-1}$, mean)",
    ]

    results_tv = []
    results_kl = []

    i = 0
    for metric in metrics:
        wrf_vals = data_pull(metric, rho=1)

        for model, expert, vals in zip(models, experts, wrf_vals):
            # TV rank correlation (index 0)
            tau_tv, pval_tv = kendalltau(expert, vals[0])
            results_tv.append({'Metric': titles[i], 'Model': model, 'Kendall τ': tau_tv, 'p-value': pval_tv})

            # KL rank correlation (index 1)
            tau_kl, pval_kl = kendalltau(expert, vals[1])
            results_kl.append({'Metric': titles[i+1], 'Model': model, 'Kendall τ': tau_kl, 'p-value': pval_kl})

        i += 2

    # Convert to DataFrames
    corr_df_tv = pd.DataFrame(results_tv)
    corr_df_kl = pd.DataFrame(results_kl)

    # Heatmap plotting function
    def plot_corr_heatmap(corr_df, title, save_title):
        plt.figure(figsize=(6, 6), dpi=200)
        heatmap_data = corr_df.pivot(index='Metric', columns='Model', values='Kendall τ')
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                    cbar_kws={'label': 'Kendall τ'}, linewidths=0.5, linecolor='black', fmt=".2f")
        plt.title(title)
        plt.ylabel("")
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(save_title)

    # Plot both
    plot_corr_heatmap(corr_df_tv, "Kendall Rank Correlation (TV)", save_title="acrossmodel_tv_kendall_heatmap.pdf")
    plot_corr_heatmap(corr_df_kl, "Kendall Rank Correlation (KL)", save_title="acrossmodel_kl_kendall_heatmap.pdf")

if __name__ == "__main__":
    # plot_other_lines_plots()
    # plot_ranking_bars()
    # plot_ranking_spearmans_overdays()
    plot_ranking_spearmans_acrossmodels()
    # plot_ranking_kendalls_acrossmodels()