import os
import pandas as pd
from .. import logger
from ..entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    def get_data(self):
        """
        Load and return the rating and movie data
        """
        try:
            data_path = self.config.data_path
            
            ratings_file = None
            movies_file = None
            
            # Search in current directory and subdirectories
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if 'ratings' in file.lower() and file.endswith('.csv'):
                        ratings_file = os.path.join(root, file)
                    elif 'movies' in file.lower() and file.endswith('.csv'):
                        movies_file = os.path.join(root, file)
            
            if ratings_file:
                ratings = pd.read_csv(ratings_file)
                logger.info(f"Loaded ratings data from {ratings_file}: {ratings.shape}")
            else:
                logger.warning(f"Ratings file not found in {data_path}")
                ratings = None
            
            if movies_file:
                movies = pd.read_csv(movies_file)
                logger.info(f"Loaded movies data from {movies_file}: {movies.shape}")
            else:
                logger.warning(f"Movies file not found in {data_path}")
                movies = None
            
            return ratings, movies

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise e

    
    def transform(self):
        """
        Perform data transformation
        """
        try:
            ratings, movies = self.get_data()
            
            if ratings is not None and movies is not None:
                # Creating a newId for every movie to reduce the range of existing movieId
                movies["newId"] = range(1, movies["movieId"].nunique()+1)

                # Converting the the UTC timestamp to Datetime
                ratings["timestamp"] = (pd.to_datetime(ratings["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d"))

                # Merge data
                data = pd.merge(ratings, movies, on='movieId', how='left')
                logger.info(f"Merged data shape: {data.shape}")
                
                # Renaming the timestamp to date
                data.rename(columns={"timestamp": "date"}, inplace=True)
                ratings.rename(columns={"timestamp": "date"}, inplace=True)

                # Updating the movieId with the newId
                data["movieId"] = data["newId"]
                movies["movieId"] = movies["newId"] 

                # Dropping the newId from the datasets
                data.drop(["newId"], axis=1, inplace=True)
                movies.drop(["newId"], axis=1, inplace=True)

                # Sorting ratings based on date
                data.sort_values(by = "date", inplace = True)
                data.reset_index(drop=True, inplace=True)            
                
                # Save transformed data
                output_path = os.path.join(self.config.root_dir, 'data.csv')
                movies_path = os.path.join(self.config.root_dir, 'movies.csv')
                ratings_path = os.path.join(self.config.root_dir, 'ratings.csv')
                os.makedirs(self.config.root_dir, exist_ok=True)
                data.to_csv(output_path, index=False)
                movies.to_csv(movies_path, index=False)
                ratings.to_csv(ratings_path, index=False)
                logger.info(f"Transformed data saved to: {output_path}")
                logger.info(f"Transformed movies saved to: {movies_path}")
                logger.info(f"Transformed ratings saved to: {ratings_path}")
            else:
                logger.warning("Cannot transform data: missing ratings or movies")

        except Exception as e:
            logger.error(f"Error during transformation: {str(e)}")
            raise e