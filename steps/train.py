import pandas as pd
import logging
from zenml import step
from src.models import LinearRegressionModel
from sklearn.base import RegressorMixin
from config.config import ModelConfig


@step
def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        config: ModelConfig
) -> RegressorMixin:
    """
    Train model
    :param model_name: Name of the model to be used
    :param x_train: Training data
    :param y_train: Training labels
    :param config: Model config
    """
    logging.info("Training model step")
    config.model_name = model_name
    try:
        # Select model
        model = globals()[config.model_name]()

        trained_model = model.train(x_train, y_train)
        logging.info("Training model step is finished")
        return trained_model
    except Exception as ex:
        logging.error(f"Failed to train the model: {ex}")
        raise ex
