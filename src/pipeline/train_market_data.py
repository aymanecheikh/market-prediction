from kfp.dsl import component, Input, Output, Dataset, Model
from kfp.compiler import Compiler


@component(
    packages_to_install=['pandas', 'scikit-learn==1.7'],
    base_image='python:3.13'
)
def train_market_data(
    dataset_X: Input[Dataset],
    dataset_y: Input[Dataset],
    model: Output[Model]
):
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import pickle

    X_train = pd.read_csv(dataset_X.path + '.csv')
    y_train = pd.read_csv(dataset_y.path + '.csv')

    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    model.metadata['framework'] = 'LR'
    file_name = model.path + f'.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(model_lr, file)


compiler = Compiler()
compiler.compile(
    pipeline_func=train_market_data,
    package_path='src/pipeline/components/train_market_data.yaml'
)