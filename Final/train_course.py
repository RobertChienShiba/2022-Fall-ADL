import lightgbm as lgb
import pandas as pd
import numpy as np
from tqdm import tqdm

from preprocess import load_data

data = load_data()

def create_user2courses(row, d, test=False):
    past_record = d[row.name].copy() if row.name in d else set()
    if not test: 
        past_record.update(set(row["course_id"].split()))
    return past_record 


data['train2courses'] = data['train'].set_index("user_id")["course_id"].apply(lambda s: set(s.split())).to_dict()
data['val_seen2courses'] = data['val_seen'].set_index("user_id").apply(lambda row: create_user2courses(row, data['train2courses']), axis=1).to_dict()

data['val_unseen2courses'] = data['val_unseen'].set_index("user_id")["course_id"].apply(lambda s: s.split()).to_dict()

data['test_seen2courses'] = data['test_seen'].set_index("user_id").apply(lambda row: create_user2courses(row, data['val_seen2courses'], test=True), axis=1).to_dict()
data['test_unseen2courses'] = data['test_unseen'].set_index("user_id").apply(lambda row: create_user2courses(row, data['val_unseen2courses'], test=True), axis=1).to_dict()
data['user2idx'] = data['users'].set_index("user_id").assign(index = list(range(len(data['users']))))["index"].to_dict()

data['users']['recreation_names_lst'] = data['users'].recreation_names.apply(lambda s: s.split(",") if isinstance(s, str) else [])

recreation_names = set()

for e in data['users']['recreation_names_lst'].to_list():
    recreation_names.update(e)

data['users']['occupation_titles_lst'] = data['users']['occupation_titles'].apply(lambda s: s.split(",") if isinstance(s, str) else [])

occupation_titles = set()

for e in data['users']['occupation_titles_lst'].to_list():
    occupation_titles.update(e)

data['users']['interests_lst'] = data['users']['interests'].apply(lambda s: s.split(",") if isinstance(s, str) else [])
interests = set()

for e in data['users']['interests_lst'].to_list():
    interests.update(e)

data['users'] = data['users'].join(pd.get_dummies(data['users']['gender']))

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(sparse_output=True)

def one_hot_encode(df, col):
    tmp_df = pd.DataFrame.sparse.from_spmatrix(
                    mlb.fit_transform(df[col]),
                    index=df.index,
                    columns=mlb.classes_)
    df = df.join(tmp_df)
    return df

# data['users'] = one_hot_encode(data['users'], 'recreation_names_lst')
# data['users'] = one_hot_encode(data['users'], 'occupation_titles_lst')
data['users'] = one_hot_encode(data['users'], 'interests_lst')

drop_cols = ["gender", "occupation_titles", "interests", "recreation_names", "recreation_names_lst", "occupation_titles_lst", "interests_lst"]

data['users_v1'] = data['users'].drop(columns=drop_cols).set_index("user_id")


data['course2idx'] = data['courses'].set_index("course_id").assign(index = list(range(len(data['courses']))))["index"].to_dict()
data['idx2course'] = {data['course2idx'][k]: k for k in data['course2idx'].keys()}


data['courses']['groups_lst'] = data['courses']['groups'].apply(lambda s: s.split(",") if isinstance(s, str) else [])
courses_groups = set()

for e in data['courses']['groups_lst'].to_list():
    courses_groups.update(e)


data['courses']['sub_groups_lst'] = data['courses']['sub_groups'].apply(lambda s: s.split(",") if isinstance(s, str) else [])
courses_sub_groups = set()

for e in data['courses']['sub_groups_lst'].to_list():
    courses_sub_groups.update(e)


from zhconv import convert_for_mw
from text2vec import SentenceModel
from bs4 import BeautifulSoup

model = SentenceModel('shibing624/text2vec-base-chinese')

def sentence2vec(s):
    s = BeautifulSoup(s).get_text()
    s = convert_for_mw(s, 'zh-cn')
    return model.encode(s)


data['courses']['desc_embed'] = data['courses']['description'].apply(sentence2vec)

data['courses'] = one_hot_encode(data['courses'], 'groups_lst')
data['courses'] = one_hot_encode(data['courses'], 'sub_groups_lst')
data['courses']['course_published_at_local'] = pd.to_datetime(data['courses']['course_published_at_local'])
data['courses']['upload_days'] = (data['courses']['course_published_at_local'] - data['courses']['course_published_at_local'].max()).dt.days.abs()

data['courses'] = data['courses'].merge(
    data['course_chapter_items'].groupby(["course_id"]).agg(video_len = ("video_length_in_seconds", "sum")),
    on=["course_id"], how="left"
)

data['courses'] = data['courses'].merge(
    data['train']['course_id'].apply(lambda s: s.split()).explode().drop_duplicates().to_frame().assign(train = True), on=["course_id"], how="left"
).merge(
    data['val_seen'].set_index("user_id").apply(lambda row: create_user2courses(row, data['train2courses']), axis=1).explode().to_frame("course_id").drop_duplicates().assign(val_seen = True), on=["course_id"], how="left"
).merge(
    data['val_unseen'].set_index("user_id")["course_id"].apply(lambda s: s.split()).explode().to_frame("course_id").drop_duplicates().assign(val_unseen = True), on=["course_id"], how="left"
).merge(
    data['test_seen'].set_index("user_id").apply(lambda row: create_user2courses(row, data['val_seen2courses'], test=True), axis=1).explode().to_frame("course_id").drop_duplicates().assign(test_seen = True)
    , on=["course_id"], how="left"
).merge(
    data['test_unseen'].set_index("user_id").apply(lambda row: create_user2courses(row, data['val_unseen2courses'], test=True), axis=1).explode().to_frame("course_id").drop_duplicates().assign(test_unseen = True)
    , on=["course_id"], how="left"
)



drop_cols = ["course_name", "teacher_id", "teacher_intro", 
             "groups", "sub_groups", "topics", "course_published_at_local", 
             "description", "will_learn", "required_tools", "recommended_background", "target_group", "groups_lst", "sub_groups_lst",
             "train", "val_seen", "val_unseen", "test_seen", "test_unseen"]


data['courses_v1'] = data["courses"].drop(columns=drop_cols).set_index("course_id")

data['courses_v1'] = data['courses_v1'][data['courses_v1'].columns.drop(["course_price", "desc_embed", "upload_days", "video_len"]).to_list() + ["course_price", "upload_days", "video_len"]]


data['courses_v1_val_seen'] = data["courses"].query("val_seen == True").drop(columns=drop_cols).set_index("course_id")
data['courses_v1_val_seen'] = data['courses_v1_val_seen'][data['courses_v1_val_seen'].columns.drop(["course_price", "desc_embed"]).to_list() + ["course_price"]]
data['courses_v1_val_unseen'] = data["courses"].query("val_unseen == True").drop(columns=drop_cols).set_index("course_id")
data['courses_v1_val_unseen'] = data['courses_v1_val_unseen'][data['courses_v1_val_unseen'].columns.drop(["course_price", "desc_embed"]).to_list() + ["course_price"]]
data['courses_v1_test_seen'] = data["courses"].query("test_seen == True").drop(columns=drop_cols).set_index("course_id")
data['courses_v1_test_seen'] = data['courses_v1_test_seen'][data['courses_v1_test_seen'].columns.drop(["course_price", "desc_embed"]).to_list() + ["course_price"]]
data['courses_v1_test_unseen'] = data["courses"].query("test_unseen == True").drop(columns=drop_cols).set_index("course_id")
data['courses_v1_test_unseen'] = data['courses_v1_test_unseen'][data['courses_v1_test_unseen'].columns.drop(["course_price", "desc_embed"]).to_list() + ["course_price"]]



from sklearn.decomposition import PCA

desc_embed = data['courses']['desc_embed'].to_list()

desc_embed = np.array(desc_embed)

pca = PCA(n_components=100)
desc_embed_reduced = pca.fit_transform(desc_embed)

# data['selected_index'] = [data['course2idx'][c] for c in popular_courses]

course_sim = np.load("./course_sim.npy")

meta_info_len = len(data['users_v1'].columns) + 3

def make_dataset(name, remove_from=None, selected_courses=None):
    user_courses_id = []
    gid = []
    for idx, row in tqdm(data[name].iterrows()):
        if remove_from:
            if row['user_id'] in data[f'{remove_from}2courses']:
                course_set = selected_courses.difference(
                    set([data['course2idx'][i] for i in data[f'{remove_from}2courses'][row['user_id']]])
                ).copy()
            else:
                course_set = selected_courses
        else:
            course_set = selected_courses
        user_courses_id.append(list(course_set))
        gid.append(len(course_set))
    x = np.zeros((sum(gid), meta_info_len+100), dtype=np.float32)
    y = np.zeros(sum(gid), dtype=bool)
    user_arr = data['users_v1'].to_numpy()
    tmp_df = data['courses_v1']
    course_arr = tmp_df.to_numpy()[:, -3:]
    cnt = 0
    for idx, row in tqdm(data[name].iterrows()):
        user_idx = data['user2idx'][row['user_id']]
        user_course_len = gid[idx]
        x[cnt:cnt+user_course_len, 0:98] = user_arr[user_idx:user_idx+1]
        x[cnt:cnt+user_course_len, 98:101] = course_arr[user_courses_id[idx]]
        x[cnt:cnt+user_course_len, 101:] = desc_embed_reduced[user_courses_id[idx]]
        target_courses = row['course_id'].split()
        for cid in target_courses:
            cid = data['course2idx'][cid]
            if cid in user_courses_id[idx]:
                course_idx = user_courses_id[idx].index(cid)
                y[cnt+course_idx] = 1
        # # find sim
        # if sample_course_sim:
        #     last_bought_course = data['course2idx'][target_courses[-1]]
        #     if tmp_course_sim[last_bought_course].max() < 0.5:
        #         continue
        #     new_course_idx = np.argsort(tmp_course_sim[last_bought_course])[-1]
        #     y[cnt+new_course_idx] = 1
        #     user_courses_id[idx].append(new_course_idx)
        cnt += user_course_len
    return x, y, gid, user_courses_id


def sample_similar_course(name, x, y, gid, user_courses_id, sample_course_idx=676):
    tmp_course_sim = course_sim.copy()
    tmp_course_sim[:, :sample_course_idx] = 0
    cnt = 0
    for idx, row in tqdm(data[name].iterrows()):
        target_courses = row['course_id'].split()
        user_course_len = gid[idx]
        last_bought_course = data['course2idx'][target_courses[-1]]
        
        if tmp_course_sim[last_bought_course].max() < 0.45:
            continue
        mask = tmp_course_sim[last_bought_course] >= 0.45
        sort_idx = tmp_course_sim[last_bought_course].argsort()
        new_course_idxs = sort_idx[mask[sort_idx]]
        new_course_idx = np.random.choice(new_course_idxs)
        
        y[cnt+new_course_idx] = 1
        user_courses_id[idx].append(new_course_idx)
        cnt += user_course_len
    return x, y, gid, user_courses_id


X_train_origin, Y_train_origin, gid_train_origin, course_id_train_origin = make_dataset("train", None, set([data['course2idx'][i] for i in data['courses'].course_id]))

X_val_unseen, Y_val_unseen, gid_val_unseen, course_id_val_unseen = make_dataset('val_unseen', None, set([data['course2idx'][i] for i in data['courses'].query("val_unseen == True").course_id]))

X_val_seen, Y_val_seen, gid_val_seen, course_id_val_seen = make_dataset('val_seen', "train", set([data['course2idx'][i] for i in data['courses'].query("val_seen == True").course_id]))

X_test_unseen, Y_test_unseen, gid_test_unseen, course_id_test_unseen = make_dataset('test_unseen', 'val_unseen', set([data['course2idx'][i] for i in data['courses'].query("test_unseen == True").course_id]))

X_test_seen, Y_test_seen, gid_test_seen, course_id_test_seen = make_dataset('test_seen', 'val_seen', set([data['course2idx'][i] for i in data['courses'].query("test_seen == True").course_id]))




from metrics import apk

data['train2courses_id'] = data['train']["course_id"].apply(lambda s: [data['course2idx'][i] for i in s.split()]).to_dict()
data['val_seen2courses_id'] = data['val_seen']["course_id"].apply(lambda s: [data['course2idx'][i] for i in s.split()]).to_dict()
data['val_unseen2courses_id'] = data['val_unseen']["course_id"].apply(lambda s: [data['course2idx'][i] for i in s.split()]).to_dict()

def select_gid(dataset_len):
    if dataset_len == len(Y_train):
        return gid_train, course_id_train, data['train2courses_id']
    elif dataset_len == len(Y_val_seen):
        return gid_val_seen, course_id_val_seen, data['val_seen2courses_id']
    else:
        return gid_val_unseen, course_id_val_unseen, data['val_unseen2courses_id']

def _map_metric(dy_pred, dy_true):
    """An eval metric that always returns the same value"""
    metric_name = 'map50'
    gid, course_id, true_course_id = select_gid(len(dy_pred))
    cnt = 0
    ap = []
    for idx, l in enumerate(gid):
        result = np.argsort(dy_pred[cnt:cnt+l])[::-1]
        selected_course = [course_id[idx][i] for i in result]
        cnt += l
        ap.append(apk(true_course_id[idx], selected_course, 50))
    is_higher_better = True
    return metric_name, np.mean(ap), is_higher_better

def _acc_metric(dy_pred, dy_true):
    """An eval metric that always returns the same value"""
    metric_name = 'acc_100'
    total_found = 0
    total_course = 0
    dy_true = dy_true.get_label()
    gid, course_id, true_course_id = select_gid(len(dy_pred))
    cnt = 0
    for idx, l in enumerate(gid):
        y_idx = true_course_id[idx]
        result = np.argsort(dy_pred[cnt:cnt+l])[::-1][:100]
        selected_course = [course_id[idx][i] for i in result]
        for idx in selected_course:
            if idx in y_idx:
                total_found += 1
        total_course += len(y_idx)
    is_higher_better = True
    return metric_name, total_found / total_course , is_higher_better

params = {
    'objective': 'lambdarank',
    'task': 'train',
    'boosting_type': 'gbdt',
    'num_class': 1,
    'metric': {'map'},
    'num_leaves': 255,
    'learning_rate': 0.035,
    # 'subsample_freq': 0,
    'max_bin': 2047,
    'subsample_for_bin': 200000,
    # 'subsample': 1.0,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'max_depth': -1,
    'min_child_samples': 200,
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
    'early_stopping_round': 200
}

model = None
for i in range(10):
    X_train, Y_train, gid_train, course_id_train = sample_similar_course("train", X_train_origin, Y_train_origin.copy(), gid_train_origin.copy(), course_id_train_origin, sample_course_idx=676)
    lgb_train = lgb.Dataset(X_train, Y_train, group=gid_train, categorical_feature=list(range(meta_info_len-1)))
    lgb_eval = lgb.Dataset(X_val_seen, Y_val_seen, group=gid_val_seen, reference=lgb_train, categorical_feature=list(range(meta_info_len-1)))
    lgb_eval_unseen = lgb.Dataset(X_val_unseen, Y_val_unseen, group=gid_val_unseen, reference=lgb_train, categorical_feature=list(range(meta_info_len-1)))
    model = lgb.train(params, lgb_train, num_boost_round=50, valid_sets=[lgb_eval_unseen, lgb_eval, lgb_train], valid_names=["unseen", "seen", "train"], feval=_acc_metric, callbacks=[lgb.log_evaluation(period=5)], init_model=model)

