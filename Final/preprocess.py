from glob import glob
import pandas as pd

def load_data():
    data_paths = glob("./data/*")

    data = {}
    for data_path in data_paths:
        data[data_path.split("/")[-1].split(".")[0]] = pd.read_csv(data_path)
    return data

