
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_svmlight_file

import config
from metrics import gini_norm
from DataReader import FeatureDictionary, DataParser
sys.path.append("..")
from DeepFM import DeepFM

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def _extract_features(df):
    print('df:', type(df), df.shape, df.head())
    df['index_str'] = df.index.astype(str)
    df['id'] = df['id'].str.cat(df['index_str'], sep=':')
    df.drop(['index_str'], axis=1, inplace=True)
    for i in range(200):
        f_name = 'f' + str(i)
        # print(type(df[f_name]))
        df[f_name] = df[f_name].str.split(':', expand=True)[1]
        # print('df[f_name]:', type(df[f_name]), df[f_name].shape, df[f_name].head())
    print('df:', type(df), df.shape, df.head())
    return df


def _load_data():
    header = ['target', 'id']
    for i in range(200):
        header.append('f' + str(i))

    dfTrain = pd.read_csv(config.TRAIN_FILE, sep=' ', names=header)
    dfTest = pd.read_csv(config.TEST_FILE, sep=' ', names=header)
    dfTrain = _extract_features(dfTrain)
    dfTest = _extract_features(dfTest)

    print('init dfTrain:', dfTrain.shape, dfTrain.head())
    print('init dfTest:', dfTest.shape, dfTest.head())

    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices


def _load_fm_data():
    header = ['target', 'id']
    for i in range(200):
        header.append('f' + str(i))

    X_train, y_train, qid_train = load_svmlight_file(config.TRAIN_FILE, query_id=True)
    X_test, y_test, ids_test = load_svmlight_file(config.TEST_FILE, query_id=True)
    X_test.resize((X_test.shape[0], 899))
    print('train x y qid:', X_train.shape, y_train.shape, qid_train.shape)
    print('test x y qid:', X_test.shape, y_test.shape, ids_test.shape)

    dfTrain = X_train
    dfTest = X_test
    cat_features_indices = []

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)

        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()


# load data
# dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_fm_data()
print('dfTrain:', dfTrain.shape)
print('dfTest:', dfTest.shape)
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('ids_test:', len(ids_test))
print('cat_features_indices:', len(cat_features_indices), cat_features_indices)

# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))


# ------------------ DeepFM Model ------------------
# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm,
    "random_seed": config.RANDOM_SEED
}
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

# # ------------------ FM Model ------------------
# fm_params = dfm_params.copy()
# fm_params["use_deep"] = False
# y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)
#
#
# # ------------------ DNN Model ------------------
# dnn_params = dfm_params.copy()
# dnn_params["use_fm"] = False
# y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)



