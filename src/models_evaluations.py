import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract class for models evaluation
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the scores for the model
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Score
        """
        pass


class MSE(Evaluation):
    """
    Mean Squared Error evaluation strategy
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        logging.info("Calculating MSE")
        try:
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as ex:
            logging.error(f"Error while calculating MSE: {ex}")
            raise ex


class R2(Evaluation):
    """
    R2 Score evaluation strategy
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        logging.info("Calculating R2 Score")
        try:
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as ex:
            logging.error(f"Error while calculating R2 Score: {ex}")
            raise ex


class RMSE(Evaluation):
    """
    Root Mean Squared Error evaluation strategy
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        logging.info("Calculating RMSE")
        try:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as ex:
            logging.error(f"Error while calculating RMSE: {ex}")
            raise ex