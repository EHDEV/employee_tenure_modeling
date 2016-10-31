from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm


def plot_group(data, x, y, group=None,  title='', ylab='', xlab='', kind='bar', rotation=0, fontsize=10):

    sns.set(style="whitegrid")
    g = sns.factorplot(x=x, y=y, hue=group, data=data,kind=kind, size=8, palette="muted", legend=True)
    g.set_xticklabels(labels=data[x].unique(), rotation=rotation, fontsize=fontsize)
    g.set_yticklabels(fontsize=fontsize)
    g.despine(left=True)
    g.set_ylabels(ylab)
    g.set_xlabels(xlab)
    g.set(title=title)

    plt.tight_layout(pad=5)


def plot_heatmap(data, x, y, group=None,  title='', ylab='', xlab='', kind='bar', rotation=0):
    sns.set(style="whitegrid")
    cmap = cm.get_cmap('OrRd', 11)
    data_piv = data.pivot(index=x, columns=group, values=y)
    g = sns.heatmap(data_piv, robust=True, cmap=cmap)
    # g = sns.factorplot(x=x, y=y, hue=group, data=data,kind=kind, size=8, palette="muted", legend=True, )
    # g.set_xticklabels(labels=data[x].unique(), rotation=rotation, fontsize=10)
    # g.set_yticklabels(fontsize=10)
    # g.despine(left=True)
    # g.set_ylabels(ylab)
    # g.set_xlabels(xlab)
    plt.xticks(rotation=90)
    g.set(title=title)


