from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
import numpy as np
import random
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

    fig,ax = plt.subplots(1, 1, figsize=figsize)
    sns.lineplot(
        data=pd.melt(data.reset_index(), id_vars="n_NA",value_vars=data.columns),
        hue="variable",
        x="n_NA",
        y="value",
        ax=ax
    )

    ax.set_xlabel("Nombre de valeurs manquantes")
    ax.set_ylabel("Pourcentage d'examen prescrit")
    ax.set_title("% de prescription de bilans en fonction du nombre de variables manquantes")

def plot_missing_bar(X, features, figsize=(15,10)):
    """
        This function produce a bar plot of all the missings values

        Parameters
        ----------
        X: Pandas Dataframe of features
        features: [str], list of variables name
        figsize: (int, int), tuple of the figure size
    """

    fig, ax = plt.subplots(1,1, figsize=figsize)

    data = (X[features].isna()*1).mean().reset_index()
    sns.barplot(
        data=data,
        x="index",
        y=0,
        ax=ax
    )

    ax.set_title("% de valeurs manquantes par variable")
    ax.set_xlabel("Variable")
    ax.set_ylabel("% de valeurs manquantes")

def plot_correlation(X, features, figsize=(10,6)):
    """
        This function produce a heatmap plot of all variables correlation values

        Parameters
        ----------
        X: Pandas Dataframe of features
        features: [str], list of variables name
        figsize: (int, int), tuple of the figure size
    """

    fig, ax = plt.subplots(figsize = figsize)

    correlation_matrix = X[features].corr()
    sns.heatmap(
        correlation_matrix,
        cmap='YlGn',
        ax=ax
    )
    
    ax.set_title('Corrélations entre les features');


def plot_labels_frequencies_and_correlation(y, labels, figsize=(30,10)):
    """
        This function produce a bar of label proportion and heatmap plot of all labels correlation values

        Parameters
        ----------
        y: Pandas Dataframe of labels
        labels: [str], list of labels name
        figsize: (int, int), tuple of the figure size
    """

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs = axs.flatten()

    # Plotting labels proportion
    labels_data = ((y[labels].sum()/y.shape[0])*100).reset_index().round(2)
    sns.barplot(
        data=labels_data,
        x="index",
        y=0,
        ax=axs[0]
    )
    axs[0].tick_params(labelrotation=45)
    axs[0].set_ylim(0,100)
    axs[0].set_title("Proportion d'examens biologiques réalisés")
    axs[0].set_xlabel("Examens biologiques")
    axs[0].set_ylabel("% d'examens réalisés")

    # Plotting correlation

    correlation_data = y[labels].corr()
    sns.heatmap(correlation_data, ax=axs[1], cmap='YlGn')
    axs[1].set_title('Correlations entre les labels');

def plot_box_variable_label_distribution(X, y, features, labels):
    """
        This function produce a box plot of the features distribution according to the variable status

        Parameters
        ----------
        X: Pandas Dataframe of features
        y: Pandas Dataframe of labels
        features: [str], list of variables name
        labels: [str], list of output name
    """

    # Generating colormap
    colors = sns.color_palette("muted", 2*len(features))

    # Getting Xy dataframe
    Xy = get_Xy_df(X, y)

    fig = plt.figure(constrained_layout=True, figsize=(5*len(labels),2*len(features)))
    figs = fig.subfigures(len(labels), 1)
    axs = [x.subplots(1, len(features)) for x in figs]

    for i in range(len(labels)):
        figs[i].suptitle(f"Distribution des variables selon le statut {labels[i]} (réalisé (1) ou non (0))")
        for j in range(len(features)):
            feature_name, variable_name = features[j], labels[i]
            axs[i][j].set_title(feature_name)
            axs[i][j].set_xlabel(variable_name)
            axs[i][j].set_ylabel(feature_name)
            sns.boxplot(
                data=Xy, 
                ax=axs[i][j], 
                x=variable_name, 
                y=feature_name, 
                showfliers=False,
                palette=colors[j*2:(j+1)*2]
            )

    fig.suptitle("Distribution des features en fonction du label")
    plt.show()

def plot_odd_word_wc(X, y, text_column, labels, min_occurrence=3, ncols=5):
    """
        This function produce a word cloud of words odd-ratio (odd-ratio of seing the word given the label)

        Parameters
        ----------
        X: Pandas Dataframe of features
        y: Pandas Dataframe of labels
        text_column: str, name of the column containing the text
        labels: [str], list of output name
        min_occurrence: int, minimum number of ocurrence of the word
        ncols: int, number of columns in the output plot
    """

    # Computing nrows an getting the structure
    nrows = len(labels)//ncols + 1*((len(labels)%ncols) != 0)
    fig = plt.figure(constrained_layout=True, figsize=(4*ncols, 5*nrows))
    figs = fig.subfigures(nrows, ncols)
    figs = figs.flatten()
    axs = [x.subplots(2, 1) for x in figs]

    def rand_color_label0(*args, **kwargs):
        return "rgb(0, 100, {})".format(random.randint(200, 455))

    def rand_color_label1(*args, **kwargs):
        return "rgb({}, 0, 100)".format(random.randint(200, 455))

    color_fn = [rand_color_label0, rand_color_label1]

    # Getting Xy
    Xy = get_Xy_df(X, y)

    # Text preprocessing
    Xy = Xy.dropna(subset=[text_column])
    Xy["text_preprocessed"] = Xy[text_column] \
        .replace(",", " ").str.lower()

    # Generating the plots
    for i in range(len(labels)):
        label = labels[i]
        figs[i].suptitle(label)

        # Filtering text data
        text_data = Xy[[label, "chiefcomplaint"]].dropna().groupby(label).agg(lambda x: " ".join(x))["chiefcomplaint"]

        # Training countvectorizer model then counting the odd
        cv = CountVectorizer().fit(text_data)
        text_data_array = (cv.transform(text_data).toarray()+1) # Smoothing count
        text_data_array[:,np.where(text_data_array <= (min_occurrence+1))[1]] = 1 # Set the odds to neutral odd
        text_data_array = text_data_array/text_data_array.sum(axis=1).reshape(2, -1) 

        for j, text in text_data.items():
            values = (text_data_array[j,:]/(text_data_array[1-j,:])).tolist()

            axs[i][j].imshow(
                WordCloud(background_color = "white", relative_scaling=0.2, max_words = 25, color_func=color_fn[j]).generate_from_frequencies(
                    frequencies=dict(zip(
                        cv.get_feature_names(),
                        values
                    ))
                )
            )
            axs[i][j].set_xlabel(f"{j}")

    fig.suptitle("WordCloud selon le label")
    plt.show()