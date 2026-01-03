import sys
from pathlib import Path
from src.Movie_Recommendation import logger
sys.path.append(str(Path(__file__).parent / 'src'))
from src.Movie_Recommendation.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.Movie_Recommendation.pipeline.stage_02_data_validation import DataValidationPipeline
from src.Movie_Recommendation.pipeline.stage_03_data_transformation import DataTransformationPipeline
from src.Movie_Recommendation.pipeline.stage_04_feature_engineering import FeatureEngineeringPipeline


STAGE_NAME = "Data Ingestion"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   pipeline = DataIngestionPipeline()
   pipeline.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e


STAGE_NAME = "Data Validation"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   pipeline = DataValidationPipeline()
   pipeline.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e


STAGE_NAME = "Data Transformation"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   pipeline = DataTransformationPipeline()
   pipeline.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e


STAGE_NAME = "Feature Engineering"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   pipeline = FeatureEngineeringPipeline()
   pipeline.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e