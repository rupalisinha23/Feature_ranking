import os.path
import warnings
import random
import pandas as pd
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings("ignore")
random.seed(1234)


class FeaturePreparer:
    """
    preprocess the data
    """
    def __init__(self, path_to_data):
        self.path = path_to_data
        self.categorical = None


    def read_data(self) -> pd.DataFrame:
        """
        Check the extention of the file to read the dataset.
        It can be extended to read xlsx, tsv or other file formats.
        :return: data in dataframe format
        """
        if self.path.split('.')[-1] == 'csv':
            return pd.read_csv(self.path)


    def seperate_variable_types(self, df) -> None:
        """
        method to check if the features are categorical or numerical.
        can be extended to temporal and other datatypes as well.
        :param df: dataframe
        :return: None
        """
        # find categorical variables
        self.categorical = [var for var in df.columns if df[var].dtype == 'O']
        print('There is {} categorical variable.'.format(len(self.categorical)))


        # find numerical variables
        numerical = [var for var in df.columns if df[var].dtype != 'O']
        print('There are {} numerical variables.'.format(len(numerical)))


    def prepare_data(self,df) -> pd.DataFrame:
        """
        the preprocessing steps have been directly taken from the .ipynb file
        the method check the null values and fill them with mean.
        :param df: input data
        :return: preprocessed data
        """
        df = df.drop(['sample index'], axis=1)
        print('Columns present: ', list(df.columns))

        # fills NA in all required variables with the mean
        count = 0
        for col in df.columns:
            if df.loc[:, (col)].isnull().mean() > 0:
                # get the mean value from the training data, i.e., self.raw_data
                mean_val = df.loc[:, (col)].mean()
                # replace it in the data to be passed to the model, i.e., self.prepared_data
                df[col].fillna(mean_val, inplace=True)
            else:
                count +=1
        if count == len(df.columns):
            print('There are no null values present in data.')

        return df


class ModelManager:
    """
    defines, runs the models and generate results
    """
    def __init__(self,FeaturePreparer, input_path, output_path):
        self.feature = FeaturePreparer(input_path)
        self.output_path = output_path


    def split_data(self, X, y) -> pd.DataFrame:
        """
        split the data into training and test set.
        :param X: dataframe
        :param y: dataframe
        :return: split dataframe
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
        return X_train, X_test, y_train, y_test


    def filter_method(self, X, y) -> pd.DataFrame:
        """
        chi2 method is used to rank the features.
        Advantages:
        Disadvantages:
        Scalability:
        :param X: feature variables
        :param y: target variable
        :return: sorted list of sensors with the most important sensor on top
        """
        best_features = SelectKBest(score_func=chi2, k="all")
        fit = best_features.fit(X, y)
        df_scores = pd.DataFrame(fit.scores_)
        df_columns = pd.DataFrame(X.columns)

        feature_scores = pd.concat([df_columns, df_scores], axis=1)
        feature_scores.columns = ['sensor', 'score']
        print('Filter method complete wth chi2 correlation test.')

        return feature_scores.sort_values(by='score', ascending=False).drop(['score'], axis=1).reset_index(drop=True)


    def wrapper_method(self, X_train, y_train, df_columns) -> pd.DataFrame:
        """
        Logistic Regression with RFE is used to rank the features.
        :param X_train: feature variables
        :param y_train: target variable
        :return: sorted list of sensors with the most important sensor on top
        """
        classifier = LogisticRegression(random_state=0)
        rfe = RFE(classifier, n_features_to_select=1)
        rfe = rfe.fit(X_train, y_train)

        classifier = LogisticRegression(random_state=0)  # can also use regularization parameter here
        classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)
        scores = []
        num_features = len(df_columns)  # excluding the target variable
        for i in range(num_features):
            scores.append((rfe.ranking_[i], df_columns[i]))
        scores.sort()

        print('Wrapper method complete using Logistic Regression with RFE.')

        return pd.DataFrame(scores, columns=['rank', 'sensor']).drop(['rank'], axis=1)


    def embedding_method(self, X_train, y_train, X_columns) -> pd.DataFrame:
        """
        Extra Tree Classifier is used to rank the features.
        Advantages:
        Disadvantages:
        Scalability:
        :param X_train: feature variables
        :param y_train: target variable
        :param X_columns: list of feature names
        :return: sorted list of sensors with the most important sensor on top
        """
        model = ExtraTreesClassifier(random_state=1234)
        model.fit(X_train, y_train)
        feature_importances = pd.concat([pd.DataFrame(model.feature_importances_),pd.DataFrame(X_columns)],axis=1)
        feature_importances.columns=['score','sensor']
        print('Embedding method complete with Extra Tree Classifier.')

        return feature_importances.sort_values(by='score', ascending=False).drop(['score'], axis=1).reset_index(drop=True)


    def generate_csv_results(self, filter_df, wrapper_df, embedding_df) -> None:
        """
        combines the results from filter, wrapper and embedding method to write to a csv file
        :param filter_df:
        :param wrapper_df:
        :param embedding_df:
        :return: None
        """
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)

        results = pd.concat([filter_df, wrapper_df, embedding_df], axis=1, ignore_index=True)
        results.columns=['chi2','logistic_regression_rfe','extra_tree_classifier']
        results.to_csv(r'./output/results.csv', index=None, header=True)

        print('Results.csv generated. Please check the output folder.')


    def run_pipeline(self):
        """
        runs the entire custom made pipeline to rank the features according to their importance
        :return: generated csv file with results
        """
        df = self.feature.read_data()
        self.feature.seperate_variable_types(df)
        df = self.feature.prepare_data(df)

        X = df.drop(['class_label'], axis = 1)
        y = df.class_label
        filter_results = self.filter_method(X,y)

        X_train, X_test, y_train, y_test = self.split_data(X, y)
        wrapper_results = self.wrapper_method(X_train, y_train, X.columns)
        embedding_results = self.embedding_method(X_train, y_train, X.columns)

        return self.generate_csv_results(filter_results, wrapper_results, embedding_results)


model = ModelManager(FeaturePreparer,'./data/task_data.csv','./output/results.csv')
model.run_pipeline()
