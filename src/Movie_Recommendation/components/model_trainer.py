import os
import numpy
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime
from .. import logger
from ..entity.config_entity import ModelTrainerConfig
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise import KNNBaseline
from surprise.model_selection import GridSearchCV
from surprise import SlopeOne
from surprise import SVD
from surprise import SVDpp
import random
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def get_data(self):
        """
        Load and return the train_regression and test_regression datasets.
        """
        try:
            data_path = self.config.data_path
            
            train_regression_file = os.path.join(data_path, "Training_Data_For_Regression.csv")
            test_regression_file = os.path.join(data_path, "Testing_Data_For_Regression.csv")
            
            if os.path.exists(train_regression_file):
                train_regression = pd.read_csv(train_regression_file)
                logger.info(f"Loaded train_regression data from {train_regression_file}: {train_regression.shape}")
            else:
                logger.warning(f"Train regression file not found in {data_path}")
                train_regression = None

            if os.path.exists(test_regression_file):
                test_regression = pd.read_csv(test_regression_file)
                logger.info(f"Loaded test_regression data from {test_regression_file}: {test_regression.shape}")
            else:
                logger.warning(f"Test regression file not found in {data_path}")
                test_regression = None


            return train_regression, test_regression

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise e
   
    def transform_data(self):
        """
        Perform necessary data transformations.
        """
        try:
            train_regression_data, test_regression_data = self.get_data()
            if train_regression_data is None or test_regression_data is None:
                logger.warning("Train or test regression data is missing.")
                return None
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(train_regression_data[["User_ID", "Movie_ID", "Rating"]], reader)
            trainset = data.build_full_trainset()
            testset = list(zip(test_regression_data["User_ID"].values, test_regression_data["Movie_ID"].values, test_regression_data["Rating"].values))
            logger.info(f"Data transformation completed.")
            return trainset , testset , data
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise e
        
    error_cols = ["Model", "Train RMSE", "Train MAPE", "Test RMSE", "Test MAPE"]
    error_table = pd.DataFrame(columns = error_cols)

    # Function to save modelling results in a table

    def make_table(self, model_name, rmse_train, mape_train, rmse_test, mape_test):
        error_cols = ["Model", "Train RMSE", "Train MAPE", "Test RMSE", "Test MAPE"]
        global error_table
        error_table = pd.concat([error_table, pd.DataFrame([[model_name, rmse_train, mape_train, rmse_test, mape_test]], columns = error_cols)])
        #error_table = error_table.append(pd.DataFrame([[model_name, rmse_train, mape_train, rmse_test, mape_test]], columns = error_cols))
        error_table.reset_index(drop = True, inplace = True)
    
    # Function to calulate RMSE and MAPE values

    def error_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(abs((y_true - y_pred)/y_true))*100
        return rmse, mape

    def plot_importance(self, model, clf):

        sns.set(style="darkgrid")
        fig = plt.figure(figsize = (25, 5))
        ax = fig.add_axes([0, 0, 1, 1])
    
        model.plot_importance(clf, ax = ax, height = 0.3)
        plt.xlabel("F Score", fontsize = 20)
        plt.ylabel("Features", fontsize = 20)
        plt.title("Feature Importance", fontsize = 20)
        plt.tick_params(labelsize = 15)
        plt.show()
    
    def train_test_xgboost(self,x_train, x_test, y_train, y_test, model_name):
        startTime = datetime.now()
        train_result = dict()
        test_result = dict()
    
        clf = xgb.XGBRegressor(n_estimators = 100, silent = False, n_jobs  = -1)
        clf.fit(x_train, y_train)
    
        print("-" * 50)
        print("TRAIN DATA")
        y_pred_train = clf.predict(x_train)
        rmse_train, mape_train = self.error_metrics(y_train, y_pred_train)
        print("RMSE : {}".format(rmse_train))
        print("MAPE : {}".format(mape_train))
        train_result = {"RMSE": rmse_train, "MAPE": mape_train, "Prediction": y_pred_train}
    
        print("-" * 50)
        print("TEST DATA")
        y_pred_test = clf.predict(x_test)
        rmse_test, mape_test = self.error_metrics(y_test, y_pred_test)
        print("RMSE : {}".format(rmse_test))
        print("MAPE : {}".format(mape_test))
        test_result = {"RMSE": rmse_test, "MAPE": mape_test, "Prediction": y_pred_test}
        
        print("-"*50)
        print("Time Taken : ", datetime.now() - startTime)
    
        self.plot_importance(xgb, clf)
        self.make_table(model_name, rmse_train, mape_train, rmse_test, mape_test)
    
        return train_result, test_result
    
    # in surprise prediction of every data point is returned as dictionary like this:
    # "user: 196        item: 302        r_ui = 4.00   est = 4.06   {'actual_k': 40, 'was_impossible': False}"
    # In this dictionary, "r_ui" is a key for actual rating and "est" is a key for predicted rating

    def get_ratings(self,predictions):
        actual = np.array([pred.r_ui for pred in predictions])
        predicted = np.array([pred.est for pred in predictions])
        return actual, predicted

    def get_error(self,predictions):
        actual, predicted = self.get_ratings(predictions)
        rmse = np.sqrt(mean_squared_error(actual, predicted)) 
        mape = np.mean(abs((actual - predicted)/actual))*100
        return rmse, mape   
     

    my_seed = 15
    random.seed(my_seed)
    np.random.seed(my_seed)

    # Running Surprise model algorithms
    def run_surprise(self, algo, trainset, testset, model_name):

        startTime = datetime.now()
    
        train = dict()
        test = dict()
    
        algo.fit(trainset)
    
        #-----------------Evaluating Train Data------------------#
        print("-"*50)
        print("TRAIN DATA")
        train_pred = algo.test(trainset.build_testset())
        train_actual, train_predicted = self.get_ratings(train_pred)
        train_rmse, train_mape = self.get_error(train_pred)
        print("RMSE = {}".format(train_rmse))
        print("MAPE = {}".format(train_mape))
        train = {"RMSE": train_rmse, "MAPE": train_mape, "Prediction": train_predicted}
    
        #-----------------Evaluating Test Data------------------#
        print("-"*50)
        print("TEST DATA")
        test_pred = algo.test(testset)
        test_actual, test_predicted = self.get_ratings(test_pred)
        test_rmse, test_mape = self.get_error(test_pred)
        print("RMSE = {}".format(test_rmse))
        print("MAPE = {}".format(test_mape))
        test = {"RMSE": test_rmse, "MAPE": test_mape, "Prediction": test_predicted}

        print("-"*50)    
        print("Time Taken = "+str(datetime.now() - startTime))
    
        self.make_table(model_name, train_rmse, train_mape, test_rmse, test_mape)
    
        return train, test

    def create_train_test_split(self, test_size=0.2):
        train_regression_data, test_regression_data = self.get_data()
        x_train = train_regression_data.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
        x_test = test_regression_data.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
        y_train = train_regression_data["Rating"]
        y_test = test_regression_data["Rating"]
        return x_train, x_test, y_train, y_test

    def build_model(self):
        x_train, x_test, y_train, y_test = self.create_train_test_split()
        model_train_evaluation = dict()
        model_test_evaluation = dict()


        logger.info("Starting model training process.")

        # Training the Xgboost Regression Model on with the 13 features
        train_result, test_result = self.train_test_xgboost(x_train, x_test, y_train, y_test, "XGBoost_13")

        model_train_evaluation["XGBoost_13"] = train_result
        model_test_evaluation["XGBoost_13"] = test_result
        
        # Applying BaselineOnly from the surprise library to predict the ratings
        bsl_options = {"method":"sgd", "learning_rate":0.01, "n_epochs":25}

        algo = BaselineOnly(bsl_options=bsl_options)
        trainset, testset, data = self.transform_data()
        train_result, test_result = self.run_surprise(algo, trainset, testset, "BaselineOnly")

        model_train_evaluation["BaselineOnly"] = train_result
        model_test_evaluation["BaselineOnly"] = test_result
        train_regression_data, test_regression_data = self.get_data()
        train_regression_data["BaselineOnly"] = model_train_evaluation["BaselineOnly"]["Prediction"]
        test_regression_data["BaselineOnly"] = model_test_evaluation["BaselineOnly"]["Prediction"]

        # Fitting the Xgboost again with new BaselineOnly feature

        x_train = train_regression_data.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
        x_test = test_regression_data.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
        y_train = train_regression_data["Rating"]
        y_test = test_regression_data["Rating"]

        train_result, test_result = self.train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL")

        model_train_evaluation["XGB_BSL"] = train_result
        model_test_evaluation["XGB_BSL"] = test_result

        # Finding the suitable parameter for Surprise KNN-Baseline with User-User Similarity

        param_grid  = {'sim_options':{'name': ["pearson_baseline"], "user_based": [True], "min_support": [2], "shrinkage": [60, 80, 80, 140]}, 'k': [5, 20, 40, 80]}
        gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(data)

        # best RMSE score
        print(gs.best_score['rmse'])
        # combination of parameters that gave the best RMSE score
        print(gs.best_params['rmse'])

        # Applying the KNN-Baseline with the searched parameters

        sim_options = {'name':'pearson_baseline', 'user_based':True, 'min_support':2, 'shrinkage':gs.best_params['rmse']['sim_options']['shrinkage']}

        bsl_options = {'method': 'sgd'} 
        
        algo = KNNBaseline(k = gs.best_params['rmse']['k'], sim_options = sim_options, bsl_options=bsl_options)
        
        train_result, test_result = self.run_surprise(algo, trainset, testset, "KNNBaseline_User")

        model_train_evaluation["KNNBaseline_User"] = train_result
        model_test_evaluation["KNNBaseline_User"] = test_result

        # Similarly finding best parameters for Surprise KNN-Baseline with Item-Item Similarity

        param_grid  = {'sim_options':{'name': ["pearson_baseline"], "user_based": [False], "min_support": [2], "shrinkage": [60, 80, 80, 140]}, 'k': [5, 20, 40, 80]}

        gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=3)

        gs.fit(data)

        # best RMSE score
        print(gs.best_score['rmse'])
        
        # combination of parameters that gave the best RMSE score
        print(gs.best_params['rmse'])

        # Applying KNN-Baseline with best parameters searched

        sim_options = {'name':'pearson_baseline', 'user_based':False, 'min_support':2, 'shrinkage':gs.best_params['rmse']['sim_options']['shrinkage']}

        bsl_options = {'method': 'sgd'} 

        algo = KNNBaseline(k = gs.best_params['rmse']['k'], sim_options = sim_options, bsl_options=bsl_options)
        
        train_result, test_result = self.run_surprise(algo, trainset, testset, "KNNBaseline_Item")

        model_train_evaluation["KNNBaseline_Item"] = train_result
        model_test_evaluation["KNNBaseline_Item"] = test_result

        # Addding the KNNBaseline features to the train and test dataset

        train_regression_data["KNNBaseline_User"] = model_train_evaluation["KNNBaseline_User"]["Prediction"]
        train_regression_data["KNNBaseline_Item"] = model_train_evaluation["KNNBaseline_Item"]["Prediction"]

        test_regression_data["KNNBaseline_User"] = model_test_evaluation["KNNBaseline_User"]["Prediction"]
        test_regression_data["KNNBaseline_Item"] = model_test_evaluation["KNNBaseline_Item"]["Prediction"]
       
        # Applying Xgboost with the KNN-Baseline newly added features

        x_train = train_regression_data.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
        x_test = test_regression_data.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

        y_train = train_regression_data["Rating"]
        y_test = test_regression_data["Rating"]

        train_result, test_result = self.train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL_KNN")

        model_train_evaluation["XGB_BSL_KNN"] = train_result
        model_test_evaluation["XGB_BSL_KNN"] = test_result

        # Appling the SlopeOne algorithm from the Surprise library
        so = SlopeOne()

        train_result, test_result = self.run_surprise(so, trainset, testset, "SlopeOne")

        model_train_evaluation["SlopeOne"] = train_result
        model_test_evaluation["SlopeOne"] = test_result  

        # Adding the SlopOne predictions to the train and test datasets

        train_regression_data["SlopeOne"] = model_train_evaluation["SlopeOne"]["Prediction"]
        train_regression_data["SlopeOne"] = model_train_evaluation["SlopeOne"]["Prediction"]

        test_regression_data["SlopeOne"] = model_test_evaluation["SlopeOne"]["Prediction"]
        test_regression_data["SlopeOne"] = model_test_evaluation["SlopeOne"]["Prediction"]

        # Matrix Factorization using SVD from Surprise Library

        # here, n_factors is the equivalent to dimension 'd' when matrix 'A'
        # is broken into 'b' and 'c'. So, matrix 'A' will be of dimension n*m. So, matrices 'b' and 'c' will be of dimension n*d and m*d.
        param_grid  = {'n_factors': [5,7,10,15,20,25,35,50,70,90]}   
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(data)

        # best RMSE score
        print(gs.best_score['rmse'])

        # combination of parameters that gave the best RMSE score
        print(gs.best_params['rmse'])

        # Applying SVD with best parameters

        algo = SVD(n_factors = gs.best_params['rmse']['n_factors'], biased=True, verbose=True)

        train_result, test_result = self.run_surprise(algo, trainset, testset, "SVD")

        model_train_evaluation["SVD"] = train_result
        model_test_evaluation["SVD"] = test_result

        # Matrix Factorization SVDpp with implicit feedback

        # Hyper-parameter optimization for SVDpp
        param_grid = {'n_factors': [10, 30, 50, 80, 100], 'lr_all': [0.002, 0.006, 0.018, 0.054, 0.10]}
        gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(data)

        # best RMSE score
        print(gs.best_score['rmse'])

        # combination of parameters that gave the best RMSE score
        print(gs.best_params['rmse'])

        #Applying SVDpp with best parameters¶



        model_train_evaluation["SVDpp"] = train_result
        algo = SVDpp(n_factors = gs.best_params['rmse']['n_factors'], lr_all = gs.best_params['rmse']["lr_all"], verbose=True)
        model_test_evaluation["SVDpp"] = test_result
        train_result, test_result = self.run_surprise(algo, trainset, testset, "SVDpp")

        # XGBoost 13 Features + Surprise BaselineOnly + Surprise KNN Baseline + SVD + SVDpp

        train_regression_data["SVD"] = model_train_evaluation["SVD"]["Prediction"]
        train_regression_data["SVDpp"] = model_train_evaluation["SVDpp"]["Prediction"]

        test_regression_data["SVD"] = model_test_evaluation["SVD"]["Prediction"]
        test_regression_data["SVDpp"] = model_test_evaluation["SVDpp"]["Prediction"]
        test_regression_data.head()
        # Applying Xgboost on the feature set

        x_train = train_regression_data.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)
        x_test = test_regression_data.drop(["User_ID", "Movie_ID", "Rating"], axis = 1)

        y_train = train_regression_data["Rating"]
        y_test = test_regression_data["Rating"]

        train_result, test_result = self.train_test_xgboost(x_train, x_test, y_train, y_test, "XGB_BSL_KNN_MF")

        model_train_evaluation["XGB_BSL_KNN_MF"] = train_result
        model_test_evaluation["XGB_BSL_KNN_MF"] = test_result

        # Applying Xgboost with Surprise's BaselineOnly + KNN Baseline + SVD + SVDpp + SlopeOne

        x_train = train_regression_data[["BaselineOnly", "KNNBaseline_User", "KNNBaseline_Item", "SVD", "SVDpp", "SlopeOne"]]
        x_test = test_regression_data[["BaselineOnly", "KNNBaseline_User", "KNNBaseline_Item", "SVD", "SVDpp", "SlopeOne"]]

        y_train = train_regression_data["Rating"]
        y_test = test_regression_data["Rating"]

        train_result, test_result = self.rain_test_xgboost(x_train, x_test, y_train, y_test, "XGB_KNN_MF_SO")

        model_train_evaluation["XGB_KNN_MF_SO"] = train_result
        model_test_evaluation["XGB_KNN_MF_SO"] = test_result
        # Visualizing the errors of all the models we tested out

        error_table2 = error_table.drop(["Train MAPE", "Test MAPE"], axis = 1)
        error_table2.plot(x = "Model", kind = "bar", figsize = (25, 8), grid = True, fontsize = 15)
        plt.title("Train and Test RMSE  of all Models", fontsize = 20)
        plt.ylabel("Error Values", fontsize = 10)
        plt.xticks(rotation=60)
        plt.legend(bbox_to_anchor=(1, 1), fontsize = 10)
        plt.show()

        logger.info("Model training completed.")


    def train(self):

        # Creating instance of svd_pp
        trainset, testset , Data = self.transform_data()
        n_factors = self.config.n_factors
        lr_all = self.config.lr_all
        verbose = self.config.verbose
        svd_pp = SVDpp(n_factors = n_factors, lr_all = lr_all, verbose=True)
        svd_pp.fit(trainset)
        predictions = svd_pp.test(trainset.build_testset())
        print(predictions)
        rmse, mape = self.get_error(predictions)
        print(f"Train RMSE: {rmse}, Train MAPE: {mape}")
        
        joblib.dump(svd_pp, self.config.trained_model_path)
        logger.info(f"Trained model saved at {self.config.trained_model_path}")
        logger.info("Training process completed.")
        
