## End to End Movie Recommendation System

### Configuration Workflow

To modify the pipeline:

1. Update `config/config.yaml` - Set paths and parameters
2. Update `params.yaml` - Modify hyperparameters
3. Update `entity/config_entity.py` - Define configuration entities
4. Update `config/configuration.py` - Implement configuration manager
5. Update components in `components/` - Modify pipeline stages
6. Update pipeline stages in `pipeline/` - Update pipeline logic
7. Update `main.py` - Execute the pipeline
8. Update `dvc.yaml` - Define DVC pipeline stages