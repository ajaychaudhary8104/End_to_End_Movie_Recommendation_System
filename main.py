import sys
from pathlib import Path
from src.Movie_Recommendation import logger
sys.path.append(str(Path(__file__).parent / 'src'))
from src.Movie_Recommendation.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.Movie_Recommendation.pipeline.stage_02_data_validation import DataValidationPipeline


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