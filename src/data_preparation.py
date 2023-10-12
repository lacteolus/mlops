import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class for handling data strategy
    """
    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[
        pd.DataFrame,
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for pre-processing data
    """
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process data
        :param df: Raw data
        :return: Output data
        """
        try:
            # Drop some columns
            df = df.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                    "customer_zip_code_prefix",
                    "order_item_id"
                ],
                axis=1
            )
            # Fill-in gaps in data
            df["product_weight_g"].fillna(df["product_weight_g"].median(), inplace=True)
            df["product_length_cm"].fillna(df["product_length_cm"].median(), inplace=True)
            df["product_height_cm"].fillna(df["product_height_cm"].median(), inplace=True)
            df["product_width_cm"].fillna(df["product_width_cm"].median(), inplace=True)
            df["review_comment_message"].fillna("No review", inplace=True)
            # Remove non-numerical columns
            df = df.select_dtypes(include=[np.number])
            return df
        except Exception as ex:
            logging.error(f"Error processing data: {ex}")
            raise ex


class DataSplitStrategy(DataStrategy):
    """
    Strategy to split data into train and test
    """
    def handle_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test
        :param df: Data
        :return:
        """
        try:
            x = df.drop(["review_score"], axis=1)
            y = df["review_score"]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            return x_train, x_test, y_train, y_test
        except Exception as ex:
            logging.error(f"Error while splitting data: {ex}")
            raise ex


class DataHandler:
    """
    Class for handling data
    """
    def __init__(self, df: pd.DataFrame, strategy: DataStrategy):
        """
        :param df: Data to handle
        :param strategy: Selected strategy to handle data
        """
        self.df = df
        self.strategy = strategy

    def handle_data(self) -> Union[
        pd.DataFrame,
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ]:
        """
        Handle data with selected strategy
        :return: Processed data
        """
        try:
            return self.strategy.handle_data(self.df)
        except Exception as ex:
            logging.error(f"Error processing data: {ex}")
            raise ex
