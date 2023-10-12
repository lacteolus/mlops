"""
Models classes
"""
import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class BaseModel(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, x_train, y_train) -> None:
        """
        Train the model
        :param x_train: Training data
        :param y_train: Training labels
        :return: None
        """
        pass


class LinearRegressionModel(BaseModel):
    """
    Linear regression model
    """
    def train(self, x_train, y_train, **kwargs) -> LinearRegression:
        """
        Train the model
        :param x_train: Training data
        :param y_train: Training labels
        :return: LinearRegression model
        """
        logging.info("Training Linear Regression Model")
        try:
            model = LinearRegression(**kwargs)
            model.fit(x_train, y_train)
            return model
        except Exception as ex:
            logging.error(f"Training failed: {ex}")
            raise ex
