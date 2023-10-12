import pandas as pd
import logging
from zenml import step


class IngestData:
    """
    Class to ingest data from data file
    """
    def __init__(self, data_path: str):
        """
        :param data_path: Path to the data
        """
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        """
        Get data from CSV file
        :return: Ingested data
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Step for ingesting data
    :param data_path: Path to the data
    :return: Ingested data
    """
    try:
        return IngestData(data_path).get_data()
    except Exception as ex:
        logging.error(f"Error while ingesting data {ex}")
        raise ex
