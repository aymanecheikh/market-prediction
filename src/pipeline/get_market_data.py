from kfp.dsl import component, Output, Dataset
from kfp.compiler import Compiler


@component(
    packages_to_install=['yfinance==0.2', 'scikit-learn==1.7'],
    base_image='python:3.13'
)
def get_market_data(
    ticker: str,
    dataset_X_train: Output[Dataset],
    dataset_y_train: Output[Dataset],
    dataset_X_test: Output[Dataset],
    dataset_y_test: Output[Dataset]
):
    import yfinance as yf
    from sklearn.model_selection import train_test_split

    df_market = yf.Ticker(
        ticker=ticker
    ).history(
        period='max'
    ).reset_index()
    df_market['SMA_50'] = df_market.Close.rolling(window=50).mean()
    df_market['SMA_200'] = df_market.Close.rolling(window=200).mean()
    df_market.dropna(inplace=True)
    X = df_market[[
        'Date', 'Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200'
    ]]
    y = df_market.Close
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    X_train.to_csv(dataset_X_train.path + '.csv', index=False)
    y_train.to_csv(dataset_y_train.path + '.csv', index=False)
    X_test.to_csv(dataset_X_test.path + '.csv', index=False)
    y_test.to_csv(dataset_y_test.path + '.csv', index=False)


compiler = Compiler()
compiler.compile(
    pipeline_func=get_market_data,
    package_path=(
        'src/pipeline/components/get_market_data.yaml'
    )
)