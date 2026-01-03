from ..config.configuration import ConfigurationManager
from ..components.feature_engineering import FeatureEngineering
from .. import logger


STAGE_NAME = "stage Feature Engineering "


class FeatureEngineeringPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feature_engineering_config = config.get_feature_engineering_config()
        feature_engineering = FeatureEngineering(config=feature_engineering_config)
        feature_engineering.GetSimilarMoviesUsingMovieMovieSimilarity("superman",10)
        feature_engineering.Calculate_User_User_Similarity(138493,10)
        feature_engineering.getUser_UserSimilarity()
        feature_engineering.extract_features()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = FeatureEngineeringPipeline()
        pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e