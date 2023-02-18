import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('./result_table.csv')

names = {
    'Baseline': 'SGD',
    'BN': 'BatchNorm',
    'SM 1000': 'ShiftMatch 1K',
#     'SM 5000': 'ShiftMatch 5K',
    'SM 10000': 'ShiftMatch 10K',
    'SM 100000': 'ShiftMatch 100K',
    'SM (Full)': 'ShiftMatch 1,280K (Full training set)'
}

palette = ['#ffffb2','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#b10026'][::-2][:2]+ [
    '#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b'
][-5:]


sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(16, 3.5))
ax = axes[0]
# df['name'] = [names[s] for s in df['method'] if s in names else s]
df["name"] = df["method"].apply(lambda s: names[s] if s in names else s)
dd = df.groupby(['name', 'level'])['acc'].mean().reset_index()
sns.barplot(data=dd,
            x='level', y='acc', hue='name', 
            palette=palette,
            hue_order = names.values(),
            ax=ax
           )
ax.get_legend().remove()
# ax.grid(True)
ax.set_ylabel('Accuracy', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('Corruption Intensity', fontsize=16)
ax.set_ylim(bottom=0.2)

ax = axes[1]
dd = df.groupby(['name', 'level'])['ll'].mean().reset_index()
sns.barplot(data=dd,
            x='level', y='ll', hue='name',
            palette=palette,
            hue_order = names.values(),
            ax=ax,
           )
handles = ax.get_legend_handles_labels()
ax.get_legend().remove()
# # ax.grid(True)
ax.set_ylabel('Log-Likelihood', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('Corruption Intensity', fontsize=16)
# figlegend = plt.figure(figsize=(2,2))
legend = fig.legend(*handles, 
#            loc ='upper left',
           loc ='lower center',
           bbox_to_anchor=(0.5, -0.3),
           fontsize=15,
           ncol=3
          )
ax.set_ylim(bottom=-4)
legend.get_frame().set_linewidth(2)
legend.get_frame().set_edgecolor("k")
# plt.axis("off")
plt.savefig("imagnet_c_result.pdf", bbox_inches="tight")
plt.savefig("imagnet_c_result.png", bbox_inches="tight")