import pandas as pd
import numpy as np

def act_preprocesser(fname: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a locally generated ATC/ACT sequences CSV and return (X, Y).
    """
    dataset_as_dataframe = pd.read_csv(fname)

    all_column_names_as_set = set(dataset_as_dataframe.columns)

    def feature_column_name_filter(column_name: str):
        try:
            return int(column_name.split("_")[-1]) < 32
        except ValueError:
            return False
        
    def target_column_name_filter(column_name: str):
        try:
            return int(column_name.split("_")[-1]) >= 32 and "position" in column_name
        except ValueError:
            return False
        
    sorting_key_function = lambda x: (ord(x[0]) - 96) * 100000 + (ord(x.split("_")[-2][-1]) - 96) * 100 + int(x.split("_")[-1])

    feature_columns_as_iterable = filter(feature_column_name_filter, all_column_names_as_set)
    target_columns_as_iterable = filter(target_column_name_filter, all_column_names_as_set)

    feature_columns_as_list = sorted(feature_columns_as_iterable, key=sorting_key_function)
    target_columns_as_list = sorted(target_columns_as_iterable, key=sorting_key_function)

    features_as_dataframe = dataset_as_dataframe[feature_columns_as_list]
    targets_as_dataframe = dataset_as_dataframe[target_columns_as_list]

    X = features_as_dataframe.to_numpy()
    Y = targets_as_dataframe.to_numpy()

    return X, Y