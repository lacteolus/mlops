# MLOps with ZenML

Creates ML pipelines

## Usage
1. Use `python run.py` to start pipeline

```
usage: run.py [-h] [--model {LinearRegressionModel}]

options:
  -h, --help            show this help message and exit
  --model {LinearRegressionModel}, -m {LinearRegressionModel}
                        Model to be used
```

2. Use `zenml up --blocking` to start ZenML server locally and navigate to  http://127.0.0.1:8237 to open ZenML Web Dashboard.