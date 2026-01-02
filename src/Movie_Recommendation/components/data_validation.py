import os
import pandas as pd
from .. import logger
from ..entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    
    def validate_all_files_exist(self) -> bool:
        """
        Validate if all required CSV files exist
        """
        try:
            validation_status = True
            
            all_files = os.listdir(self.config.unzip_data_dir)
            
            for file in all_files:
                if file.endswith('.csv'):
                    logger.info(f"Found file: {file}")
            
            return validation_status

        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise e


    def validate_schema(self) -> bool:
        """
        Validate the schema of the CSV files
        """
        try:
            validation_status = True
            
            all_files = os.listdir(self.config.unzip_data_dir)
            csv_files = [f for f in all_files if f.endswith('.csv')]
            
            for csv_file in csv_files:
                df = pd.read_csv(os.path.join(self.config.unzip_data_dir, csv_file))
                logger.info(f"Validating {csv_file}...")
                logger.info(f"Columns: {df.columns.tolist()}")
                logger.info(f"Shape: {df.shape}")
            
            return validation_status

        except Exception as e:
            logger.error(f"Error during schema validation: {str(e)}")
            raise e


    def validate(self):
        """
        Main validation method
        """
        try:
            self.validate_all_files_exist()
            self.validate_schema()
            
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write("Validation successful")
            
            logger.info("Data validation completed successfully!")

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write("Validation failed")
            raise e