from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from .preprocessing import get_Xy_df

def plot_all_scatter (X, variables, ncols=3, figsize=(20,10)):
    """
        This function produce a scatter view of all the variables from a dataset

        Parameters
        ----------
        X: Pandas Dataframe
        variables: [str], list of variables name
        n_cols: int, number of columns in the plot
        figsize: (int, int), tuple of the figure size
    """

    # Getting nrows
    nrows = (len(variables) // ncols) + 1*((len(variables) % ncols) != 0)

    figs, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()

    for i in range(len(variables)):
        variable = variables[i]
        sns.scatterplot(
            data=X[variable].value_counts(),
            ax = axs[i]
        )

        axs[i].ticklabel_format(style='scientific', axis='x', scilimits=(0, 4))
        axs[i].set_xlabel("Valeur")
        axs[i].set_ylabel("Nombre d'occurences")
        axs[i].set_title(variable)

    plt.tight_layout()

def plot_missing_outcome(X, y, features, labels, figsize=(20,10)):
    """
        This function produce a line plot of all the missings values according to the outcomes values

        Parameters
        ----------
        X: Pandas Dataframe of features
        y: Pandas Dataframe of labels
        features: [str], list of variables name
        labels: [str], list of output name
        figsize: (int, int), tuple of the figure size
    """

    Xy = get_Xy_df(X, y)
    data = Xy[labels].join(
            pd.DataFrame(Xy[features].isna().astype("int").sum(axis=1))
        ).rename(columns={0:"n_NA"}) \
        .groupby("n_NA") \
        .agg(lambda x: x.sum()/x.count())

    fig,ax = plt.subplots(1, 1, figsize=(20,10))
    sns.lineplot(
        data=pd.melt(data.reset_index(), id_vars="n_NA",value_vars=data.columns),
        hue="variable",
        x="n_NA",
        y="value",
        ax=ax
    )

    ax.set_xlabel("Nombre de valeurs manquantes")
    ax.set_ylabel("Pourcentage d'examen prescrit")