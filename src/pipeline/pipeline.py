from typing import NamedTuple
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
    Artifact
)


@component(
    packages_to_install=['yfinance, scikit-learn'],
    base_image='python:3.13'
)
def get_market_data(ticker: str, output_csv: Output[Dataset]):
    
    import yfinance as yf
    
    df = yf.Ticker(ticker).history(period='max').reset_index()
    df['SMA_50'] = df.Close.rolling(window=50).mean()
    df['SMA_200'] = df.Close.rolling(window=200).mean()
    df.dropna(inplace=True)
    df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200']]
    df.to_csv(output_csv.path, index=False)


@component(
    packages_to_install=['pandas', 'scikit-learn'],
    base_image='python:3.13'
)
def preprocess_data(
    input_csv: Input[Dataset],
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    output_ytrain: Output[Dataset],
    output_ytest: Output[Dataset]
):
    
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(input_csv.path)
    features = df.drop(columns=['Close'])
    target = df.Close
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features,
        target,
        test_size=0.2,
        random_state=42
    )
    pd.DataFrame(X_train).to_csv(output_train.path, index=False)
    pd.DataFrame(X_test).to_csv(output_test.path, index=False)
    pd.DataFrame(y_train).to_csv(output_ytrain.path, index=False)
    pd.DataFrame(y_test).to_csv(output_ytest.path, index=False)


@component(
    packages_to_install=['pandas', 'scikit-learn'],
    base_image='python:3.13'
)
def train_model(
    train_data: Input[Dataset],
    ytrain_data: Input[Dataset],
    model_output: Output[Model],
):
    
    import pickle
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    
    X_train = pd.read_csv(train_data.path)
    y_train = pd.read_csv(ytrain_data.path).values.ravel()
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_output.metadata['framework'] = 'LR'
    file_name = model_output + '.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)


@component(
    packages_to_install=['pandas', 'scikit-learn'],
    base_image='python:3.13'
)
def evaluate_model(
    test_data: Input[Dataset],
    ytest_data: Input[Dataset],
    model: Input[Model],
    thresholds_dict_str: str,
    metrics_output: Output[Dataset],
    kpi: Output[Metrics]
) -> NamedTuple('output', [('deploy', str)]): # type: ignore
    
    import logging
    import typing
    import pickle
    import json
    import pandas as pd
    from sklearn.metrics import accuracy_score
    
    def threshold_check(val):
        cond = 'false'
        if val > 0.8:
            cond = 'true'
        return cond

    X_test = pd.read_csv(test_data.path)
    y_test = pd.read_csv(ytest_data.path)
    file_name = model.path + '.pkl'
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    thresholds_dict = json.loads(thresholds_dict_str)
    model.metadata['accuracy'] = float(accuracy)
    kpi.log_metric('accuracy', float(accuracy))
    deploy = threshold_check(float(accuracy))
    return (deploy,)


@component(
    packages_to_install=['google-cloud-aiplatform', 'scikit-learn', 'kfp'],
    base_image='python:3.13',
    output_component_file='model_marketdata_component.yml'
)
def deploy_marketdata_prediction(
    model: Input[Model],
    project: str,
    region: str,
    serving_container_image_uri: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model]
):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)

    DISPLAY_NAME = 'marketdataprediction'
    MODEL_NAME = 'marketdataprediction_lr'
    ENDPOINT_NAME = 'marketdataprediction_endpoint'

    def create_endpoint():
        endpoints = aiplatform.Endpoint.list(
            filter='display_name="{}"'.format(ENDPOINT_NAME),
            order_by='create_time desc',
            project=project,
            location=region
        )
        if len(endpoints) > 0:
            endpoint = endpoints[0]
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=ENDPOINT_NAME, project=project, location=region
            )
    endpoint = create_endpoint()

    model_upload = aiplatform.Model.upload(
        display_name = DISPLAY_NAME,
        artifact_uri= model.uri.replace('model', ''),
        serving_container_image_uri = serving_container_image_uri,
        serving_container_health_route=f'/v1/models/{MODEL_NAME}',
        serving_container_predict_route=f'/v1/models/{MODEL_NAME}:predict',
        serving_container_environment_variables={
            'MODEL_NAME': MODEL_NAME,
        },
    )

    model_deploy = model_upload.deploy(
        machine_type='n1-standard-4',
        endpoint=endpoint,
        traffic_split={'0': 100},
        deployed_model_display_name=DISPLAY_NAME,
    )

    vertex_model.uri = model_deploy.resource_name