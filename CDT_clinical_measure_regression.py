import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import MaxNLocator

path = 'E:/2022_ECG_inference/Cardiac_Personalisation/'
data = pd.read_csv(path + 'MI_inference_results_without_MIcompactness_ori.csv')
my_pal_two = {'Scar': '#288596', "BZ": '#7D9083'}
my_pal_two_new = {'Subendo': '#F7F6EE', "Transmu": '#FADA95'}
# my_pal_two_new = {'Subendo': '#62C890', "Transmu": '#D7F0BC'}
# my_pal_two_new = {'Subendo': '#9FDDED', "Transmu": '#2A90AB'}
scar_color, BZ_color = my_pal_two['Scar'], my_pal_two['BZ']
scanario_name_1 = ['A1_subendo', 'A2_subendo', 'A3_subendo', 'A4_subendo', 'B1_large_subendo', 'B1_small_subendo', 'B2_subendo', 'B3_subendo']
scanario_name_2 = ['A1_transmural', 'A2_transmural', 'A3_transmural', 'A4_transmural', 'B1_large_transmural', 'B1_small_transmural', 'B2_transmural', 'B3_transmural']
scanario_name_list = scanario_name_1 + scanario_name_2
scanario_name_list_MIsize = ['A2_10_20_transmural', 'A2_15_25_transmural', 'A2_20_30_transmural', 'A2_25_35_transmural', 'A2_transmural', 'A2_35_45_transmural']
# MIsize_mapping = {'A2_10_20_transmural': 1, 'A2_15_25_transmural': 2, 'A2_25_35_transmural': 3, 'A2_transmural': 4, 'A2_35_45_transmural': 5} 
MIsize_mapping = {'A2_10_20_transmural': 0.10, 'A2_15_25_transmural': 0.15, 'A2_20_30_transmural': 0.20, 'A2_25_35_transmural': 0.25, 'A2_transmural': 0.30, 'A2_35_45_transmural': 0.35} 
recon_type_list = ['recon_geo', 'recon_ECG']
MItype_mapping_subendo = {'A1_subendo': 1, 'A2_subendo': 2, 'A3_subendo': 3, 'A4_subendo': 4, 'B1_large_subendo': 5, 'B1_small_subendo': 6, 'B2_subendo': 7, 'B3_subendo': 8} 
MItype_mapping_transmu = {'A1_transmural': 1, 'A2_transmural': 2, 'A3_transmural': 3, 'A4_transmural': 4, 'B1_large_transmural': 5, 'B1_small_transmural': 6, 'B2_transmural': 7, 'B3_transmural': 8} 
scanario_name_12 = ['A1_subendo', 'A1_transmural', 'A2_subendo', 'A2_transmural', 'A3_subendo', 'A3_transmural', 'A4_subendo', 'A4_transmural', 'B1_large_subendo', 'B1_large_transmural', 'B1_small_subendo', 'B1_small_transmural', 'B2_subendo', 'B2_transmural', 'B3_subendo', 'B3_transmural']
MItype_mapping_12 = {'A1_subendo': 1, 'A1_transmural': 2, 'A2_subendo': 3, 'A2_transmural': 4, 'A3_subendo': 5, 'A3_transmural': 6, 'A4_subendo': 7, 'A4_transmural': 8, 'B1_large_subendo': 9, 'B1_large_transmural': 10, 'B1_small_subendo': 11, 'B1_small_transmural': 12, 'B2_subendo': 13, 'B2_transmural': 14, 'B3_subendo': 15, 'B3_transmural': 16}

def ECG_rename(ECG_name):
    ECG_name = ECG_name.replace('A1', 'Septal').replace('A2', 'Apical').replace('A3', 'Ext anterior').replace('A4', 'Lim anterior')
    ECG_name = ECG_name.replace('B1', 'Lateral').replace('B2', 'Inferior').replace('B3', 'Inferolateral')
    # ECG_name = ECG_name.replace('normal', 'Baseline')
    # ECG_name = ECG_name.replace('transmural', 'transmu')
    ECG_name = ECG_name.replace('_transmural', '')
    ECG_name = ECG_name.replace('_subendo', '')

    return ECG_name

def draw_scatterplot(pre_scar, gd_scar, pre_BZ, gd_BZ, ax):
    slope, intercept, r_value, p_value, std_err = stats.linregress(gd_scar, pre_scar)
    xfit = np.linspace(min(gd_scar), max(gd_scar), 100)
    yfit = slope * xfit + intercept
    ax.scatter(gd_scar, pre_scar, c=scar_color, marker='o', alpha=0.9)
    ax.text(0.95, 0.98, s= 'r$^2$ = %.3f' % (r_value**2), fontsize='small', alpha=0.7, ha='right', va='top', transform=ax.transAxes)
    ax.plot(xfit, yfit, c=scar_color)

    slope, intercept, r_value, p_value, std_err = stats.linregress(gd_BZ, pre_BZ)
    xfit = np.linspace(min(gd_BZ), max(gd_BZ), 100)
    yfit = slope * xfit + intercept
    ax.scatter(gd_BZ, pre_BZ, c=BZ_color, marker='s', alpha=0.9)
    ax.text(0.95, 0.9, s= 'r$^2$ = %.3f' % (r_value**2), fontsize='small', alpha=0.7, ha='right', va='top', transform=ax.transAxes)
    ax.plot(xfit, yfit, c=BZ_color)

    return

def draw_MIsize_boxplot(pre_scar_Dice, pre_BZ_Dice):

    df1 = pd.DataFrame.from_dict(pre_scar_Dice)
    df2 = pd.DataFrame.from_dict(pre_BZ_Dice)

    # 创建一个空的DataFrame，用于存储所有数据
    df = pd.DataFrame(columns=['MIsize', 'Dice', 'Category'])

    # 添加第一类数据，设置Category为'scar'，并调整MIsize的位置
    df1['Category'] = 'Scar'
    df1 = df1.melt(id_vars='Category', var_name='MIsize', value_name='Dice')
    df1['MIsize'] = df1['MIsize'].astype(str)  # 将子类标签转换为字符串类型
    # df1['MIsize'] = df1['MIsize'].apply(lambda x: '    ' + x)  # 调整位置，添加空格
    df = pd.concat([df, df1])

    # 添加第二类数据，设置Category为'BZ'，并调整MIsize的位置
    df2['Category'] = 'BZ'
    df2 = df2.melt(id_vars='Category', var_name='MIsize', value_name='Dice')
    df2['MIsize'] = df2['MIsize'].astype(str)  # 将子类标签转换为字符串类型
    # df2['MIsize'] = df2['MIsize'].apply(lambda x: '    ' + x)  # 调整位置，添加空格
    df = pd.concat([df, df2])

    df['MIsize'] = df['MIsize'].map(MIsize_mapping)
    ax = sns.boxplot(data=df, x='Dice', y='MIsize', hue='Category', palette=my_pal_two, linewidth=1, orient="h", saturation=1)
    ax.invert_yaxis()  # 反转 y 轴坐标

    plt.xlabel('Dice score') 
    plt.ylabel('Size of post-MI (radius of scars)')

    # 调整x轴标签的显示角度
    # plt.xticks(rotation=45, ha='right')

    plt.savefig("fig_Dice_ApicalMIsize.pdf", format="pdf")
    plt.show()

def draw_MIDice_boxplot(pre_scar_Dice, pre_BZ_Dice, MItype_mapping, MI_extent):
    # fig, ax = plt.subplots(figsize=(6, 6))  # 调整figsize为你期望的正方形尺寸
    fig, ax = plt.subplots(figsize=(5, 4))  # 调整figsize为你期望的正方形尺寸

    df1 = pd.DataFrame.from_dict(pre_scar_Dice)
    df2 = pd.DataFrame.from_dict(pre_BZ_Dice)

    # 创建一个空的DataFrame，用于存储所有数据
    df = pd.DataFrame(columns=['MItype', 'Dice', 'Category'])

    # 添加第一类数据，设置Category为'scar'，并调整MItype的位置
    df1['Category'] = 'Scar'
    df1 = df1.melt(id_vars='Category', var_name='MItype', value_name='Dice')
    df1['MItype'] = df1['MItype'].astype(str)  # 将子类标签转换为字符串类型
    # df1['MItype'] = df1['MItype'].apply(lambda x: '    ' + x)  # 调整位置，添加空格
    df = pd.concat([df, df1])

    # 添加第二类数据，设置Category为'BZ'，并调整MItype的位置
    df2['Category'] = 'BZ'
    df2 = df2.melt(id_vars='Category', var_name='MItype', value_name='Dice')
    df2['MItype'] = df2['MItype'].astype(str)  # 将子类标签转换为字符串类型
    # df2['MItype'] = df2['MItype'].apply(lambda x: '    ' + x)  # 调整位置，添加空格
    df = pd.concat([df, df2])

    df['MItype'] = df['MItype'].map(MItype_mapping)
    ax = sns.boxplot(data=df, x='Dice', y='MItype', hue='Category', palette=my_pal_two, linewidth=1, orient="h", saturation=1)
    ax.set_xlim([-0.05, 1.05])  # Set y-axis limits
    ax.invert_yaxis()  # 反转 y 轴坐标

    plt.ylabel('MI scenario')
    plt.xlabel('Dice score') 
    
    # 调整x轴标签的显示角度
    # plt.xticks(rotation=45, ha='right')
    plt.savefig('fig_boxplot_Dice_' + MI_extent + '.pdf', format="pdf")
    plt.show()

def draw_scatterplot_recon(recon, Dice_scar, Dice_BZ, ax):
    slope, intercept, r_value, p_value, std_err = stats.linregress(recon, Dice_scar)
    xfit = np.linspace(min(recon), max(recon), 100)
    yfit = slope * xfit + intercept
    ax.scatter(recon, Dice_scar, c=scar_color, marker='o', alpha=0.9)
    ax.text(0.95, 0.98, s= 'r$^2$ = %.3f' % (r_value**2), fontsize='small', alpha=0.7, ha='right', va='top', transform=ax.transAxes)
    ax.plot(xfit, yfit, c=scar_color)

    slope, intercept, r_value, p_value, std_err = stats.linregress(recon, Dice_BZ)
    xfit = np.linspace(min(recon), max(recon), 100)
    yfit = slope * xfit + intercept
    ax.scatter(recon, Dice_BZ, c=BZ_color, marker='s', alpha=0.9)
    ax.text(0.95, 0.9, s= 'r$^2$ = %.3f' % (r_value**2), fontsize='small', alpha=0.7, ha='right', va='top', transform=ax.transAxes)
    ax.plot(xfit, yfit, c=BZ_color)

    return

def draw_MI_AHAscore_boxplot(pre_scar_AHAscore_list_subendo, pre_scar_AHAscore_list_transmu, MItype_mapping, MI_extent):
    fig, ax = plt.subplots(figsize=(5, 4))  # 调整figsize为你期望的正方形尺寸
    # fig, ax = plt.subplots(figsize=(6, 6))  # 调整figsize为你期望的正方形尺寸

    df1 = pd.DataFrame.from_dict(pre_scar_AHAscore_list_subendo)
    df2 = pd.DataFrame.from_dict(pre_scar_AHAscore_list_transmu)

    # 创建一个空的DataFrame，用于存储所有数据
    df = pd.DataFrame(columns=['MItype', 'AHA_loc_score', 'Category'])

    # 添加第一类数据，设置Category为'scar'，并调整MItype的位置
    df1['Category'] = 'Subendo'
    df1 = df1.melt(id_vars='Category', var_name='MItype', value_name='AHA_loc_score')
    df1['MItype'] = df1['MItype'].astype(str)  # 将子类标签转换为字符串类型
    df1['MItype'] = df1['MItype'].str.replace('_subendo', '')
    # df1['MItype'] = df1['MItype'].apply(lambda x: '    ' + x)  # 调整位置，添加空格
    df = pd.concat([df, df1])

    # 添加第二类数据，设置Category为'BZ'，并调整MItype的位置
    df2['Category'] = 'Transmu'
    df2 = df2.melt(id_vars='Category', var_name='MItype', value_name='AHA_loc_score')
    df2['MItype'] = df2['MItype'].astype(str)  # 将子类标签转换为字符串类型
    df2['MItype'] = df2['MItype'].str.replace('_transmural', '')
    # df2['MItype'] = df2['MItype'].apply(lambda x: '    ' + x)  # 调整位置，添加空格
    df = pd.concat([df, df2])

    # df['MItype'] = df['MItype'].map(MItype_mapping)
    
    ax = sns.boxplot(data=df, x='AHA_loc_score', y='MItype', hue='Category', palette=my_pal_two_new, linewidth=1, orient="h", saturation=1)
    ax.set_xlim([-0.05, 1.05])  # Set y-axis limits
    ax.invert_yaxis()  # 反转 y 轴坐标

    plt.ylabel('MI scenario')
    plt.xlabel('AHA-loc-score') 
    
    # 调整x轴标签的显示角度
    # plt.xticks(rotation=45, ha='right')
    plt.savefig('fig_boxplot_AHAscore_' + MI_extent + '.pdf', format="pdf")
    plt.show()

visualize_AHAscore_boxplot = False
if visualize_AHAscore_boxplot:
    pre_scar_AHAscore_list_transmu = {}
    pre_scar_AHAscore_list_subendo = {}
    for scanario_name in scanario_name_1:
        if scanario_name in data['MI_type'].values:
            pre_scar_AHAscore = data[data['MI_type'] == scanario_name]['AHA_loc_score']
            pre_scar_AHAscore_list_subendo[scanario_name] = pre_scar_AHAscore    
    for scanario_name in scanario_name_2:
        if scanario_name in data['MI_type'].values:
            pre_scar_AHAscore = data[data['MI_type'] == scanario_name]['AHA_loc_score']
            pre_scar_AHAscore_list_transmu[scanario_name] = pre_scar_AHAscore  
    draw_MI_AHAscore_boxplot(pre_scar_AHAscore_list_subendo, pre_scar_AHAscore_list_transmu, MItype_mapping_12, 'all')

visualize_MIDice_boxplot = True
if visualize_MIDice_boxplot:
    # pre_scar_Dice_list = {}
    # pre_BZ_Dice_list = {}
    # for scanario_name in scanario_name_12:
    #     if scanario_name in data['MI_type'].values:
    #         pre_scar_Dice = data[data['MI_type'] == scanario_name]['Dice_Scar']
    #         pre_BZ_Dice = data[data['MI_type'] == scanario_name]['Dice_BZ']
    #         pre_scar_Dice_list[scanario_name] = pre_scar_Dice # dict
    #         pre_BZ_Dice_list[scanario_name] = pre_BZ_Dice           
    # draw_MIDice_boxplot(pre_scar_Dice_list, pre_BZ_Dice_list, MItype_mapping_12, 'all')

    pre_scar_Dice_list = {}
    pre_BZ_Dice_list = {}
    for scanario_name in scanario_name_1:
        if scanario_name in data['MI_type'].values:
            pre_scar_Dice = data[data['MI_type'] == scanario_name]['Dice_Scar']
            pre_BZ_Dice = data[data['MI_type'] == scanario_name]['Dice_BZ']
            pre_scar_Dice_list[scanario_name] = pre_scar_Dice # dict
            pre_BZ_Dice_list[scanario_name] = pre_BZ_Dice           
    draw_MIDice_boxplot(pre_scar_Dice_list, pre_BZ_Dice_list, MItype_mapping_subendo, 'subendo')

    pre_scar_Dice_list = {}
    pre_BZ_Dice_list = {}
    for scanario_name in scanario_name_2:
        if scanario_name in data['MI_type'].values:
            pre_scar_Dice = data[data['MI_type'] == scanario_name]['Dice_Scar']
            pre_BZ_Dice = data[data['MI_type'] == scanario_name]['Dice_BZ']
            pre_scar_Dice_list[scanario_name] = pre_scar_Dice # dict
            pre_BZ_Dice_list[scanario_name] = pre_BZ_Dice           
    draw_MIDice_boxplot(pre_scar_Dice_list, pre_BZ_Dice_list, MItype_mapping_transmu, 'transmu')

visualize_MIsize_boxplot = False
if visualize_MIsize_boxplot:
    pre_scar_Dice_list = {}
    pre_BZ_Dice_list = {}
    for scanario_name in scanario_name_list_MIsize:
        if scanario_name in data['MI_type'].values:
            pre_scar_Dice = data[data['MI_type'] == scanario_name]['Dice_Scar']
            pre_BZ_Dice = data[data['MI_type'] == scanario_name]['Dice_BZ']
            pre_scar_Dice_list[scanario_name] = pre_scar_Dice # dict
            pre_BZ_Dice_list[scanario_name] = pre_BZ_Dice
            
    draw_MIsize_boxplot(pre_scar_Dice_list, pre_BZ_Dice_list)

visualize_volume_correlation = False
if visualize_volume_correlation:
    # fig, (axes) = plt.subplots(3, 6, figsize=(18, 10))
    fig, (axes) = plt.subplots(2, 8, figsize=(18, 4.2))
    axes = axes.reshape(-1)
    for ax, scanario_name in zip(axes, scanario_name_list):
        gd_scar = data[data['MI_type'] == scanario_name]['gd_MI_size_Scar']
        gd_BZ = data[data['MI_type'] == scanario_name]['gd_MI_size_BZ']
        pre_scar = data[data['MI_type'] == scanario_name]['pre_MI_size_Scar']
        pre_BZ = data[data['MI_type'] == scanario_name]['pre_MI_size_BZ']
        max_val = max(gd_scar.max(), pre_scar.max(), gd_BZ.max(), pre_BZ.max())
        ax.plot([0, max_val], [0, max_val], color='#E3DECA', linestyle='--') # #FADA95 
        draw_scatterplot(pre_scar, gd_scar, pre_BZ, gd_BZ, ax)        
        ax.set_title(ECG_rename(scanario_name))
        ax.set_xlim([0, max_val])  # Set x-axis limits
        ax.set_ylim([0, max_val])  # Set y-axis limits
        # Set the number of tick positions based on the max_val
        num_ticks = 4
        x_locator = MaxNLocator(num_ticks)
        y_locator = MaxNLocator(num_ticks)
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.axis('scaled')

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    # fig.suptitle('MI size: Prediction vs. Ground truth', x=0.5, y=0.95, fontweight='bold')
    plt.show()
    fig.savefig('fig_MIsize_regression.pdf')

visualize_recon_correlation = False
if visualize_recon_correlation:
    fig, (axes) = plt.subplots(1, 2, figsize=(10, 4))
    axes = axes.reshape(-1)
    for ax, recon_type in zip(axes, recon_type_list):
        recon = data[recon_type]
        Dice_scar = data['Dice_Scar']
        Dice_BZ = data['Dice_BZ']
        draw_scatterplot_recon(recon, Dice_scar, Dice_BZ, ax)

        ax.set_xlabel(recon_type)
        ax.set_ylabel('Dice of MI inference')
      
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.show()
    fig.savefig('fig_recon_regression.pdf')