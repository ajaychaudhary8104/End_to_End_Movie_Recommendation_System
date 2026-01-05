import os
import random
import pandas as pd
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import mlflow
import mlflow.sklearn
from .. import logger
from ..entity.config_entity import ModelEvaluationConfig
from collections import defaultdict


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def get_data(self):
        """
        Load  data
        """
        try:
            movies = pd.read_csv(self.config.movies_path)
            logger.info(f"Loaded movies data shape: {movies.shape}")
            return movies

        except Exception as e:
            logger.error(f"Error loading movies data: {str(e)}")
            raise e


    def get_ratings(self,predictions):
        actual = np.array([pred.r_ui for pred in predictions])
        predicted = np.array([pred.est for pred in predictions])
        return actual, predicted
   
    def get_error(self,predictions):
        actual, predicted = self.get_ratings(predictions)
        rmse = np.sqrt(mean_squared_error(actual, predicted)) 
        mape = np.mean(abs((actual - predicted)/actual))*100
        return rmse, mape 
    
    # Testing the recommendations made by SVDpp Algorithm

    def Get_top_n(self,predictions, n=10):

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, mid, true_r, est, _ in predictions:
            top_n[uid].append((mid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    # Print the recommended items for each user

    def Generate_Recommendated_Movies(self, u_id, predictions, n=10):

        recommend = pd.DataFrame(self.Get_top_n(predictions, n=n)[u_id], columns=["Movie_Id", "Predicted_Rating"])
        movies = self.get_data()
        recommend = recommend.merge(movies, how="inner", left_on="Movie_Id", right_on="movieId")
        recommend = recommend[["Movie_Id", "title", "genres", "Predicted_Rating"]]

        return recommend[:n]
    
    def evaluate(self):
        """
        Evaluate the trained model and log metrics to MLflow
        """
        try:
            # Load model and test data
            model = joblib.load(self.config.model_path)
            test_set = joblib.load(self.config.test_data_path)
            predictions = model.test(test_set)
            actual, predicted = self.get_ratings(predictions)
            logger.info(f"actual ratings: {actual[:10]} and predicted ratings: {predicted[:10]}")
            top_n = self.Get_top_n(predictions, n=10)
            # Saving the sampled user id list to help generate movies

            sampled_user_id = list(top_n.keys())

            # Generating recommendation using the user_Id

            test_id = random.choice(sampled_user_id)
            print("The user Id is : ", test_id)
            self.Generate_Recommendated_Movies(test_id,predictions)

            rmse, mape = self.get_error(predictions)

            if not np.isnan(rmse) and not np.isnan(mape):
                
                logger.info(f"RMSE: {rmse}")
                logger.info(f"MAPE: {mape}")
                
                # Log metrics to MLflow
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                mlflow.set_experiment(self.config.mlflow_experiment_name)
                with mlflow.start_run():
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("mape", mape)
                    mlflow.sklearn.log_model(model, "svd_recommendation_model")
                    mlflow.set_tag("model_type", "SVDpp Recommendation Model")
                    
                    # Save metrics to file
                    metrics = {"rmse": float(rmse), "mape": float(mape)}
                    os.makedirs(self.config.root_dir, exist_ok=True)
                    
                    with open(self.config.metric_file_name, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    
                    logger.info(f"Metrics saved to {self.config.metric_file_name}")
            else:
                logger.warning("No valid predictions were made")

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise e