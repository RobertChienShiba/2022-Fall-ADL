import lightgbm as lgb
import pandas as pd
import numpy as np
from tqdm import tqdm

from collections import defaultdict

from preprocess import load_data

data = load_data()

data['user2idx'] = data['users'].set_index("user_id").assign(index = list(range(len(data['users']))))["index"].to_dict()

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

X_test_seen, Y_test_seen = make_dataset("test_seen", test_seen_gid)
X_test_unseen, Y_test_unseen = make_dataset("test_unseen", test_unseen_gid)

model = lgb.Booster(model_file='subgroup_model.txt')

output = model.predict(X_test_seen)

result = defaultdict(list)
for idx, row in data['test_seen_group'].iterrows():
    result["user_id"].append(row['user_id'])
    predict_top50 = np.argsort(output[idx*91:(idx+1)*91])[::-1][:50]
    result["subgroup"].append(" ".join([str(i) for i in predict_top50]))
result_df = pd.DataFrame(result)

result_df.to_csv("test_seen_group.csv", index=False)


output = model.predict(X_test_unseen)

result = defaultdict(list)
for idx, row in data['test_unseen_group'].iterrows():
    result["user_id"].append(row['user_id'])
    predict_top50 = np.argsort(output[idx*91:(idx+1)*91])[::-1][:50]
    result["subgroup"].append(" ".join([str(i) for i in predict_top50]))
result_df = pd.DataFrame(result)

result_df.to_csv("test_unseen_group.csv", index=False)