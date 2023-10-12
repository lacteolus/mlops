from zenml import pipeline
from steps.ingest import ingest_data
from steps.prepare import prepare_data
from steps.train import train_model
from steps.evaluate import evaluate_model


@pipeline(enable_cache=True)
def training_pipeline(
        data_path: str,
        model_name: str,
) -> None:
    """
    Training pipeline
    :param model_name: Name of the model to be used
    :param data_path: Data path
    """
    raw_data = ingest_data(data_path)
    x_train, x_test, y_train, y_test = prepare_data(raw_data)
    model = train_model(
        x_train=x_train,
        y_train=y_train,
        model_name=model_name
    )
    mse, r2, rmse = evaluate_model(model, x_test, y_test)
