import os

import requests

from src.utils import create_dir


def download_data(
    url: str = "https://archive.ics.uci.edu/static/public/19/car+evaluation.zip",
    output: str = "data",
) -> str:
    response = requests.get(url, allow_redirects=True)
    create_dir(output)
    with open(os.path.join(output, "dataset.zip"), mode="wb") as file:
        file.write(response.content)
    return os.path.join(output, "dataset.zip")
