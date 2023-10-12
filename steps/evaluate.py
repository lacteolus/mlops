import pandas as pd
import logging
from zenml import step
from src.models_evaluations import RMSE, MSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated


@step
def evaluate_model(
        model: RegressorMixin,
        x_test: pd.DataFrame,
        y_test: pd.Series
) -> Tuple[
    Annotated[float, "MSE"],
    Annotated[float, "R2"],
    Annotated[float, "RMSE"]
]:
    """
    Evaluate model
    :param y_test: Test data
    :param x_test: Test labels
    :param model: Model to evaluate
    :return: Scores [mse, r2, rmse]
    """
    logging.info("Evaluating model step")
    try:
        y_pred = model.predict(x_test)
        mse = MSE().calculate_scores(y_test, y_pred)
        r2 = R2().calculate_scores(y_test, y_pred)
        rmse = RMSE().calculate_scores(y_test, y_pred)
    except Exception as ex:
        logging.error(f"Unable to evaluate model: {ex}")
        raise ex
    return mse, r2, rmse
