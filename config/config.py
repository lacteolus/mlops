from zenml.steps import BaseParameters


class ModelConfig(BaseParameters):
    """
    Model config class
    """
    model_name = "LinearRegressionModel"
