from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay, f1_score, confusion_matrix
from wordcloud import WordCloud
import seaborn as sns
import itertools
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

def vizualize_features_selection (scores, score_name, f_precision=2, n_score_max=5, ncols=3):
    """
        This function produce an heatmap of metrics score according to each variables combination

        Parameters
        ----------
        scores: Dictionnary containing a list of combination and associated score for each label produced by the .models.get_features_selection function
        score_name: str, Name of the score
        f_precision: int, floating point precision is the number of decimal to keep
        n_score_max: int, maximum number of scores to display
        ncols: int, number of columns in the output plot
    """

    # Creating a dataframe containing the scores
    scores_df = []
    for key, value in scores.items():
        scores_df_temp = pd.DataFrame(
            [dict(zip(x[0], [x[1] for i in range(len(x[0]))])) for x in value]
        ).assign(score=lambda x: x.max(axis=1))
        scores_df_temp.iloc[:,:-1] = (scores_df_temp.iloc[:,:-1].fillna("")*0).astype("str").replace("0.0", "x")
        scores_df_temp["name"] = key
        scores_df.append(scores_df_temp.sort_values("score", ascending=False))

    scores_df = pd.concat(scores_df).reset_index(drop=True)
    scores_df["n_features"] = (scores_df == "x").sum(axis=1)
    scores_df[score_name] = scores_df["score"].round(f_precision)
    scores_df = scores_df.sort_values(["name", "roc_auc", score_name], ascending=[True, False, True]).drop_duplicates(["name", score_name])

    # Plotting the dataframe
    scores_list = scores_df["name"].drop_duplicates().values.tolist()
    ncols = 3
    nrows = len(scores_list)//ncols + (len(scores_list)%ncols != 0)*1

    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols,4*nrows))
    axs = axs.flatten()

    for i in range(len(scores_list)):
        score = scores_list[i]
        sns.heatmap(
            (scores_df.query(f"name == '{score}'").set_index("roc_auc").head(n_score_max).iloc[:, :-3] == 'x')*1,
            ax=axs[i]
        )
        axs[i].set_title(score)

    fig.suptitle(f"{score_name} according to features included in the model")
    plt.tight_layout()

def display_model_performances(classifier, X_test, y_test, algorithm_name="", threshold=0.5, ncols=1):
    """
        This function produce a vizualization of the model performances

        Parameters
        ----------
        classifier: python object which should contains a predict and a predict_proba method, if many labels a dict in the format {label:classifier,...} is expected
        X_test: pandas dataframe of the features
        y_test: pandas dataframe of the labels
        algorithm_name: str, name of the algorithm
        threshold: float, threshold for classification
        ncols: int, number of columns
    """

    # Checking type of y_test
    if isinstance(y_test, pd.Series):
        y_test = pd.DataFrame(y_test)

    # Checking if one or many labels
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        if isinstance(classifier, dict) == False or len(classifier.keys()) != y_test.shape[1]:
            raise ValueError("You should provide as many classifier than labels")
    else:
        if isinstance(classifier, dict) == False:
            classifier = {y_test.columns[0]:classifier}

    labels = y_test.columns.tolist()

    # Construction of the pyplot object
    nrows = (len(labels)//ncols) + ((len(labels)%ncols)!=0)*1
    fig = plt.figure(constrained_layout=True, figsize=(15*ncols,7*nrows))
    figs = fig.subfigures(nrows, ncols)
    figs = figs.flatten()
    if len(labels) == 1:
        figs = [figs]
    axs = [x.subplots(1, 2) for x in figs]

    # For each label :
    for i in range(len(labels)):
        label = labels[i]
        label_classifier = classifier[label]
        figs[i].suptitle(label)

        y_test_true = y_test[label].values
        y_test_hat_proba = label_classifier.predict_proba(X_test)[:,1]
        y_test_hat = (y_test_hat_proba >= threshold)*1

        # Computation of metrics
        f1_score_, accuracy_score_, recall_score_, precision_score_ = [x(y_test_true, y_test_hat) for x in [f1_score, accuracy_score, recall_score, precision_score]]
        auc_score_ = roc_auc_score(y_test_true, y_test_hat_proba)
        confusion_matrix_ = confusion_matrix(y_test_true, y_test_hat)

        # Plotting
        ## Confusion matrix
        ConfusionMatrixDisplay(
            confusion_matrix_,
            display_labels=[0, 1]
        ).plot(
            ax=axs[i][0]
        )

        ## ROC curve
        fpr, tpr, thresholds = roc_curve(
            y_test_true,
            y_test_hat_proba
        )

        axs[i][1].plot(
            fpr,
            tpr,
            label=f"AUC: {auc_score_:.2f}\nF1-Score: {f1_score_:.2f}\nRecall: {recall_score_:.2f}\nPrecision: {precision_score_:.2f}\nAccuracy: {accuracy_score_:.2f}"
        )
        axs[i][1].legend(loc=4, fontsize="x-large")
        axs[i][1].set_ylabel('Taux de vrai positifs')
        axs[i][1].set_xlabel('Taux de faux positifs')

    fig.suptitle(f"Performance de l'algorithme {algorithm_name} avec un threshold de {threshold}")