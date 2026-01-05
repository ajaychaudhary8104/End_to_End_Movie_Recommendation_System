from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: str
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    test_data_path: Path
    data_path: Path
    n_factors : int
    lr_all : float
    verbose : bool

@dataclass(frozen=True)    
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    test_data_path: Path
    metric_file_name: Path
    movies_path: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str