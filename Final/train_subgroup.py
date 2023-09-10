import lightgbm as lgb
import pandas as pd
import numpy as np
from tqdm import tqdm

from preprocess import load_data

import joblib

data = load_data()

data['user2idx'] = data['users'].set_index("user_id").assign(index = list(range(len(data['users']))))["index"].to_dict()
data['train_group'] = data['train_group'].assign(
    subgroup_lst = lambda df_: df_['subgroup'].apply(lambda s: s.split() if isinstance(s, str) else []),
    target = lambda df_: df_['subgroup_lst'].apply(lambda s: [int(i) for i in s])
)

data['val_seen_group'] = data['val_seen_group'].assign(
    subgroup_lst = lambda df_: df_['subgroup'].apply(lambda s: s.split() if isinstance(s, str) else []),
    target = lambda df_: df_['subgroup_lst'].apply(lambda s: [int(i) for i in s])
)

data['test_seen_group'] = data['test_seen_group'].assign(
    subgroup_lst = lambda df_: df_['subgroup'].apply(lambda s: s.split() if isinstance(s, str) else []),
    target = lambda df_: df_['subgroup_lst'].apply(lambda s: [int(i) for i in s])
)

data['test_unseen_group'] = data['test_unseen_group'].assign(
    subgroup_lst = lambda df_: df_['subgroup'].apply(lambda s: s.split() if isinstance(s, str) else []),
    target = lambda df_: df_['subgroup_lst'].apply(lambda s: [int(i) for i in s])
)

user_embed = np.load("user_embed_onehot_v1.npy")

def create_gid(name, length):
    return [91] * length

train_gid = create_gid("train", len(data['train']))
val_seen_gid = create_gid("val_seen", len(data['val_seen']))
test_seen_gid = create_gid("test_seen", len(data['test_seen']))
test_unseen_gid = create_gid("test_unseen", len(data['test_unseen']))

def make_dataset(name, gid):
    x = np.zeros((sum(gid), user_embed.shape[1]+1), dtype=np.float32)
    y = np.zeros(sum(gid), dtype=bool)
    cnt = 0
    for idx, row in tqdm(data[f'{name}_group'].iterrows()):
        user_idx = data['user2idx'][row['user_id']]
        user_course_len = gid[idx]
        x[cnt:cnt+user_course_len, 0:98] = user_embed[user_idx:user_idx+1]
        x[cnt:cnt+user_course_len, 98] = np.arange(1, 92)
        for cid in row['target']:
            y[cnt+cid] = 1
        cnt += user_course_len
    return x, y


X_train, Y_train = make_dataset("train", train_gid)

X_val_seen, Y_val_seen = make_dataset("val_seen", val_seen_gid)

X_test_seen, Y_test_seen = make_dataset("test_seen", test_seen_gid)

X_test_unseen, Y_test_unseen = make_dataset("test_unseen", test_unseen_gid)


params = {
    'objective': 'lambdarank',
    'task': 'train',
    'boosting_type': 'gbdt',
    'num_class': 1,
    'metric': {'map'},
    'num_leaves': 255,
    'learning_rate': 0.035,
    'max_bin': 2047,
    'subsample_for_bin': 200000,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'max_depth': -1,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'force_col_wise':'true',
    'verbose': 0,
    'eval_at': 50,
    'feature_fraction': 0.8,
    'num_threads': 32,
    "first_metric_only": True,
    'early_stopping_round': 100
}

lgb_train = lgb.Dataset(X_train, Y_train, group=train_gid, categorical_feature=list(range(user_embed.shape[1])))
lgb_eval = lgb.Dataset(X_val_seen, Y_val_seen, group=val_seen_gid, reference=lgb_train, categorical_feature=list(range(user_embed.shape[1])))

model = lgb.train(params, lgb_train, num_boost_round=10000, valid_sets=[lgb_eval, lgb_train], valid_names=["seen", "train"], callbacks=[lgb.log_evaluation(period=5)])

model.save_model("subgroup_model.pkl")