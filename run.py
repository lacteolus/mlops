import os
from pipelines.training import training_pipeline
import argparse


if __name__ == "__main__":
    # Parse cmd arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model",
        "-m",
        action="store",
        dest="model_name",
        required=False,
        default="LinearRegressionModel",
        help="Model to be used",
        choices=[
            "LinearRegressionModel"
        ]
    )
    args = arg_parser.parse_args()

    # Run pipeline
    training_pipeline(
        data_path=os.path.join("data", "customers_dataset.csv"),
        model_name=args.model_name
    )
