from tqdm.auto import tqdm
import pandas as pd

from preprocess import load_data

data = load_data()

data["users"]["occupation_titles_lst"] = data["users"]["occupation_titles"].apply(
    lambda s: s.split(",") if isinstance(s, str) else []
)
data["users"]["interests_lst"] = data["users"]["interests"].apply(
    lambda s: s.split(",") if isinstance(s, str) else []
)
data["users"] = data["users"].join(pd.get_dummies(data["users"]["gender"]))


from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(sparse_output=True)


def one_hot_encode(df, col):
    tmp_df = pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(df[col]), index=df.index, columns=mlb.classes_
    )
    df = df.join(tmp_df)
    return df


data["users"] = one_hot_encode(data["users"], "interests_lst")

drop_cols = [
    "gender",
    "occupation_titles",
    "interests",
    "recreation_names",
    "occupation_titles_lst",
    "interests_lst",
]

tmp_df = data["users"].drop(columns=drop_cols).set_index("user_id")

import numpy as np

np.save("user_embed_onehot_v1.npy", tmp_df.to_numpy())
