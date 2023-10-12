import pandas as pd
import logging
from zenml import step
from src.data_preparation import DataHandler, DataPreprocessStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def prepare_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Clean and split data
    :param df: Raw data
    :return: Prepared data as tuple(x_train, x_test, y_train, y_test)
    """
    logging.info("Cleaning and splitting data")
    try:
        # Clean data
        cleaned_data = DataHandler(df, DataPreprocessStrategy()).handle_data()
        # Split data
        x_train, x_test, y_train, y_test = DataHandler(cleaned_data, DataSplitStrategy()).handle_data()
        logging.info("Data cleaning and splitting completed")
        return x_train, x_test, y_train, y_test
    except Exception as ex:
        logging.error(f"Unable to handle data: {ex}")
        raise ex
