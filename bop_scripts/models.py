from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
from .preprocessing import OutlierRemover, TextPreprocessing
from warnings import simplefilter
import itertools

def generate_model (classifier, categorical_variables, continuous_variables, text_variable=None, missing_indicator=True, OrdinalEncoder_kwargs={}, StandardScaler_kwargs={}, CountVectorizer_kwargs={}, SimpleImputer_kwargs={}, MissingIndicator_kwargs={}, remove_outliers=False, outliers_variables_ranges=None):
    """
        Generate a model Pipeline containing features pre-processing 

        Parameters
        ----------
        classifier: sklearn classifier object with a fit and transform method
        categorical_variables: [str], list of categorical variables
        continuous_variables: [str], list of continuous variables
        text_variable: [str], text variables, None if missing
        missing_indicator: boolean, if True a missing indicator is added to the Pipeline
        OrdinalEncoder_kwargs: dict, argument passed to the ordinal encoder
        StandardScaler_kwargs: dict, argument passed to the standard scaler
        CountVectorizer_kwargs: dict, argument passed to the count vectorizer
        SimpleImputer_kwargs: dict, argument passed to the simple imputer
        MissingIndicator_kwargs: dict, argument passed to the missing indicator    
        remove_outliers: boolean, if true the outliers are set to nan
        outliers_variables_ranges: Dict(variable:[range_inf, range_sup], ...), dictionnary containing for each variable the inferior and superior range
    """

    variables = categorical_variables+continuous_variables
    if text_variable is not None:
        variables += [text_variable]

    # Features pre-processing
    features_preprocessing_list = []

    ## Outliers removal :
    if remove_outliers==True and outliers_variables_ranges is not None:
        # Creating the range list
        outliers_variables_range_clean = dict([(x, y) for x,y in outliers_variables_ranges.items() if x in variables])
        features_preprocessing_list.append(("outliers", OutlierRemover(variables_ranges=outliers_variables_range_clean), list(outliers_variables_range_clean.keys())))

    if len(categorical_variables) > 0:
        features_preprocessing_list.append(("binary_encoder", OrdinalEncoder(**OrdinalEncoder_kwargs), categorical_variables))
    if len(continuous_variables) > 0:
        features_preprocessing_list.append(("continuous_scaling", StandardScaler(**StandardScaler_kwargs), continuous_variables))
    if text_variable is not None:
        # Text pre-processing then count vectorizer
        text_preprocessing_pipeline = Pipeline([
            ("text_preprocessing", TextPreprocessing()),
            ("text_countvectorizer", CountVectorizer(**CountVectorizer_kwargs))
        ])
        
        features_preprocessing_list.append(("text_encoding", text_preprocessing_pipeline, text_variable))
        
    # Imputation methods
    imputation_list = []
    imputation_list.append(("missing_imputer", SimpleImputer(**SimpleImputer_kwargs)))
    if missing_indicator:
        imputation_list.append(
            ("missing_indicator", MissingIndicator(**MissingIndicator_kwargs))
        )

    # Creating the pipeline

    features_preprocessing = ColumnTransformer(features_preprocessing_list)
    full_preprocessing = Pipeline([
        ("features", features_preprocessing),
        ("impute_and_store_missing", FeatureUnion(imputation_list)),
    ])

    pipeline = Pipeline([
        ("preprocessing", full_preprocessing),
        ("lr", classifier)
    ])
    
    return pipeline

def get_features_selection (X, y, classifier, categorical_variables, continuous_variables, text_variable=None, min_features=1, cv=3, metric_score="auc_score"):
    """
        This function return the metrics score according to each variables combination

        Parameters
        ----------
        X: Pandas Dataframe of features
        y: Pandas Dataframe of labels
        classifier: sklearn classifier object with a fit and transform method
        categorical_variables: [str], list of categorical variables
        continuous_variables: [str], list of continuous variables
        text_variable: [str], text variables, None if missing
        min_features: int, minimum number of features to include in the model
        cv: int, cross-validation splitting strategy according to the cross_val_score documentation
        metric_score: str, metric score to evaluate the model, according to sklearn.metrics.SCORERS.keys()

        Output
        ------
        Dictionnary containing a list of combination and associated score for each label
    """

    # Getting labels list
    labels = y.columns.tolist()

    # Getting the combinations
    variables = categorical_variables + continuous_variables
    if text_variable is not None:
        variables += [text_variable]

    variables_combinations = []
    for i in range(min_features, len(variables)+1):
        variables_combinations += itertools.combinations(variables, i)

    # Getting global model
    global_pipeline = generate_model(
            classifier,
            categorical_variables,
            continuous_variables,
            text_variable,
            CountVectorizer_kwargs={"ngram_range":(1,1), "max_features":200}
    )

    # Preprocessing the data : we accept here to mix train/eval in feature scaling to reduce execution time
    X_transformed = global_pipeline.steps[0][1].steps[0][1].fit_transform(X)

    # Storing scores dictionnary
    scores = dict(zip(
        labels,
        [[] for x in labels]
    ))

    # Getting the scores
    for variable_combination in variables_combinations:
        combination_categorical_variables, combination_continuous_variables = [x for x in categorical_variables if x in variable_combination], [x for x in continuous_variables if x in variable_combination]
        combination_text_variable = text_variable if (text_variable is not None and text_variable in variable_combination) else None

        pipeline = generate_model(
            classifier,
            combination_categorical_variables,
            combination_continuous_variables,
            combination_text_variable,
            CountVectorizer_kwargs={"ngram_range":(1,1), "max_features":200}
        )

        # Get X index
        if text_variable is not None:
            variables_index = [variables.index(x) for x in variable_combination if x != text_variable]
            if text_variable in variable_combination:
                variables_index += list(range(X_transformed.shape[1]-200, X_transformed.shape[1]))
        else:
            variables_index = [variables.index(x) for x in variable_combination]

        pipeline.steps[0][1].steps.pop(0) # Removing preprocessing step

        for label in labels:
            score = cross_val_score(pipeline, X_transformed[:,variables_index], y[label], cv=cv, scoring="roc_auc").mean().mean()
            scores[label].append([variable_combination, score])

    return scores

def fit_all_classifiers(classifier, X_train, y_train, hide_warnings=True):
    """
        This function fill all the models for each label.

        Parameters:
        ----------
        model: Classifier with a fit method
        X: Pandas Dataframe of features
        y: Pandas Dataframe of labels
        hide_warnings: boolean, if true the warnings will be hidden

        Output:
        -------
        Dictionnary containing a classifier per label
    """

    if hide_warnings == True:
        simplefilter("ignore", category=ConvergenceWarning)

    labels = y_train.columns.tolist()
    classifiers = {}
    for label in labels:
        classifiers[label] = classifier.fit(X_train, y_train[label])

    return classifiers