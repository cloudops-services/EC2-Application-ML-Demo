import pickle
import shutil
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder


def create_dir(dir: str):
    dir_path = Path(dir)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(exist_ok=True, parents=True)


def extract_zipfile(zipfile_path: str, output: str):
    with ZipFile(zipfile_path) as zipfile:
        zipfile.extractall(output)


def transform_data(
    buying: str,
    maint: str,
    doors: str,
    person: str,
    lug_boot: str,
    safety: str,
    dummy_columns: list[str],
) -> pd.DataFrame:
    new_row_df = pd.DataFrame(0, index=[0], columns=dummy_columns)
    new_row_df[f"buying_{buying}"] = 1
    new_row_df[f"maint_{maint}"] = 1
    new_row_df[f"doors_{doors}"] = 1
    new_row_df[f"person_{person}"] = 1
    new_row_df[f"lug_boot_{lug_boot}"] = 1
    new_row_df[f"safety_{safety}"] = 1
    # ddata = xgb.DMatrix(new_row_df)
    # return ddata
    return new_row_df


def load_model(model_path: str) -> RandomForestClassifier:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def model_predict(
    model: RandomForestClassifier, data: pd.DataFrame, y_enc: OrdinalEncoder
):
    pred = model.predict(data).reshape(-1, 1)
    return y_enc.inverse_transform(pred)[0][0]


def load_dummy_columns(dummy_columns_path: str) -> list[str]:
    with open(dummy_columns_path, "rb") as file:
        dummy_columns: list[str] = pickle.load(file)
    return dummy_columns


def load_target_enc(target_enc_path: str) -> OrdinalEncoder:
    with open(target_enc_path, "rb") as file:
        y_enc: OrdinalEncoder = pickle.load(file)
    return y_enc
