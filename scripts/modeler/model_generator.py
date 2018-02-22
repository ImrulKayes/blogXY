# Standard import
import pandas as pd
import numpy as np
import logging
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier  
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import Imputer

# Local import
import conf.config as conf
from scripts.modeler.feature_generator import ExtractFeature, ExtractBinaryFeature, ExtractLastOnlineFeature

class ModelGenerator(object):
    """ This class creates the model. It does the following:
        (1) Reads the data taking input file paths from configs
        (2) Creates the feature pipeline
        (3) Splits the dataset into training and testing, creates stratified balanced partitions for trainining data
        (4) Does a grid search a selects optimal parameters using cross-validation
        (5) Fits the model with optimal parameters and writes the model
        (6) Does performance measurement on testing dataset for the fitted model and reports results
    """
    def __init__(self):

        # Initialize variables
        self.blogster_profile_schema = conf.blogster_profile_schema
        self.blogster_profile_schema_type = conf.blogster_profile_schema_type
        self.binary_features = conf.binary_features
        self.general_features = conf.general_features
        self.initial_features = conf.initial_features
        self.profile_data = conf.profile_data
        self.model_output = conf.model_output
        self.feature_size = len(self.binary_features) + len(self.general_features) + 1

        # Set logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(conf.model_generation_log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # ExtraTreesClassifier: randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset 
        # The Extra-Trees algorithm builds an ensemble of unpruned decision or regression trees 
        # according to the classical top-down procedure. Its two main differences with other treebased ensemble methods 
        # are that it splits nodes by choosing cut-points fully at random and that it uses the whole learning sample (rather than a bootstrap replica) to grow the trees.
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.65.7485&rep=rep1&type=pdf

        # RandomForestClassifier: A random forest is a meta estimator that fits a number of decision tree classifiers 
        # on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting

        # AdaBoostClassifier: fits a classifier on the original dataset and then fits additional copies
        # of the classifier on the same dataset but where the weights of incorrectly classified instances 
        # are adjusted such that subsequent classifiers focus more on difficult cases.

        # XGBClassifier: (XGBoost) tree boosting, additive training https://xgboost.readthedocs.io/en/latest/model.html
        
        # SGDClassifier: regularized linear models with stochastic gradient descent (SGD) learning, 
        # log logistic regression, 'hinge' gives a linear SVM, 'modified_huber' brings tolerance to outliers, 'perceptron' gives the perceptron algorithm

        self.ml_models = { 
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'XGBClassifier': XGBClassifier(),
            'SGDClassifier': SGDClassifier()
        }

        # Model tuning parameters
        self.ml_model_params = { 
            'ExtraTreesClassifier': { 'n_estimators': range(10, 100, 10)},
            'RandomForestClassifier': { 'n_estimators': range(10, 100, 10)},
            'AdaBoostClassifier':  { 'n_estimators': range(10,100, 10)},
            'GradientBoostingClassifier': { 'n_estimators': range(10,100, 10), 'learning_rate': np.arange(0.1, 1.0, 0.2) },
            'SVC': 
                {'kernel': ['linear'], 'C': [1, 10]},
            'XGBClassifier': { 'n_estimators': range(10,100, 10)},
            'SGDClassifier': {'loss': ['hinge','modified_huber', 'log'], 'penalty': ['l1','l2', 'elasticnet']},
        }


    def run(self):
        self.logger.info("------------------------------------------------")
        self.logger.info("Started building models")

        # Read profile data
        df = self.read_data(self.profile_data)

        # Create dataframe from input features
        X = df[self.initial_features]

        # Create output labels
        y = df[['gender']].gender

        # Create feature pipeline
        feature_pipeline = self.create_feature_pipeline()

        # Do training-testing split and balanced training partitioning using feature pipeline and data
        (X_res, y_res, X_test, y_test) = self.split_balanced_data(X, y, feature_pipeline)
        
        # Create prediction pipelines using all different classifiers
        prediction_pipelines = self.create_prediction_pipeline()

        # Do parameter tuning, fit, test and write the best model
        self.search_fit_write(prediction_pipelines, X_res, y_res, X_test, y_test)
        
        self.logger.info("Finished building models")
        self.logger.info("------------------------------------------------")

    def read_data(self, profile_data):
        """Reads data to a dataframe from input file paths"""
        df = pd.read_csv(profile_data, sep='|', names =self.blogster_profile_schema, dtype =self.blogster_profile_schema_type, error_bad_lines=False)
        df = df[(df.gender=='Male') | (df.gender=='Female')]
        df['gender'] = df.gender.apply(lambda x: 0 if x=='Male' else 1)
        return df

    def create_feature_pipeline(self):
        """Creates feature pipeline"""
        
        # Extract all binary feature
        binary_feature_pipelines = []
        for x in self.binary_features:
            binary_pipeline = Pipeline([
                    (x,  ExtractBinaryFeature(x))
            ])
            binary_feature_pipelines.append(binary_pipeline)

        # Extract last online feature
        last_online_featurizer = Pipeline([
           ('last_online_extractor',  ExtractLastOnlineFeature('lastOnline'))
         ])

        # Extract general features
        general_feature_pipelines = []
        for x in self.general_features:
            general_pipeline = Pipeline([
                    (x,  ExtractFeature(x)),
                    (x+"scale", StandardScaler())
            ])
            general_feature_pipelines.append(general_pipeline)
    
        # Union all the features
        features = FeatureUnion(
            [(str(i), binary_pipeline) for i, binary_pipeline in enumerate(binary_feature_pipelines)] + 
            [(str(i)+"0", general_pipeline) for i, general_pipeline in enumerate(general_feature_pipelines)] + 
            [('last_online_feature', last_online_featurizer)]
        )
        return features

    def split_balanced_data(self, X, y, feature_pipeline):
        """Splits the dataset into training and testing, creates stratified balanced partitions for trainining data"""

        
        # Fit and transform the dataset
        feature_pipeline.fit(X)
        X = feature_pipeline.transform(X)

        imp = Imputer(missing_values='NaN', strategy='median', axis=0)
        X= imp.fit_transform(X)

        # Do the train-test split 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = conf.test_size, random_state=12, stratify=y)
        
        # Do a balanced training dataset 
        sm = SMOTE(random_state=42)

        X_res, y_res = sm.fit_sample(X_train, y_train)
        
        return X_res, y_res, X_test, y_test

    def create_prediction_pipeline(self):
        """Creates prediction pipeline using classifiers and feature selector"""
        
        prediction_pipelines = []
        for key in self.ml_models:
            model = self.ml_models[key]
            model_param = self.ml_model_params[key]

            estimators = [  ("imputer", Imputer(missing_values=0, strategy="median", axis=0)),
                            ('anova', SelectKBest(mutual_info_classif)), 
                            (key, model)
            ]

            pipeline_param = {}
            for param_key in model_param:
                pipeline_param["{0}__{1}".format(key, param_key)] = model_param[param_key]
            
            # Add anova parameter
            pipeline_param['anova__k'] = range(1, self.feature_size + 1)
            prediction_pipelines.append((Pipeline(estimators), pipeline_param))
            
        return prediction_pipelines


    def search_fit_write(self, fit_pipelines, X_res, y_res, X_test, y_test):
        """Does grid search, fits, and writes the model"""

        best_model = ''
        best_model_roc = float('-Inf')
        best_model_param = ''
        best_model_pipe = ''
        best_model_confusion_matrix = ''

        for (pipe, param) in fit_pipelines:
            self.logger.info("Fitting best score")
            clf = GridSearchCV(pipe, param_grid = param, cv = 10, scoring = 'roc_auc', n_jobs = -1, verbose = 2)
            clf = clf.fit(X_res, y_res)
            self.logger.info("Fitted best score")
            self.logger.info(clf.best_score_)
            self.logger.info("Fitted best params")
            self.logger.info(clf.best_params_)
            self.logger.info("Fitted best estimator")
            self.logger.info(clf.best_estimator_)

            # Do the prediction on test data
            y_pred = clf.predict(X_test)

            # Write the performance results in the logger
            self.logger.info("Test data accuracy")
            self.logger.info(accuracy_score(y_test, y_pred))
            self.logger.info("Test data ROC")
            roc_val = roc_auc_score(y_test, y_pred)
            if roc_val > best_model_roc:
                best_model = clf
                best_model_roc = roc_val
                best_model_param = param
                best_model_pipe = pipe
                best_model_confusion_matrix = confusion_matrix(y_test, y_pred)              

            self.logger.info(roc_val)
            self.logger.info("Test data confusion matrix")
            self.logger.info(confusion_matrix(y_test, y_pred))

        # Log best model stats
        self.logger.info("Best model stats")
        self.logger.info("ROC {0}".format(best_model_roc))
        self.logger.info("Confusion martix")
        self.logger.info(best_model_confusion_matrix)
        self.logger.info("Model parameter")        
        self.logger.info(best_model_param)
        self.logger.info("Pipeline")          
        self.logger.info(best_model_pipe)        

        # Write the model
        joblib.dump(best_model, self.model_output) 


if __name__ == "__main__":
    """ Create the object of the model generator and class and run"""
    model_generator = ModelGenerator()
    model_generator.run()
