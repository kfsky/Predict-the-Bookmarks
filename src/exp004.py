
import os
import warnings
import sys
import joblib
import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from time import time
from contextlib import contextmanager
from typing import Optional, Tuple
from sklearn.metrics import confusion_matrix
import datetime
import umap


sys.path.append("../")

from mypipe.config import Config
from mypipe.utils import reduce_mem_usage
from mypipe.experiment import exp_env
from mypipe.experiment.runner import Runner
from mypipe.models.model_lgbm import MyLGBMClassifier
from mypipe.Block_features import BaseBlock, ContinuousBlock, CountEncodingBlock, OneHotEncodingBlock, \
    LabelEncodingBlock, ArithmeticOperationBlock, AggregationBlock, WrapperBlock


# ---------------------------------------------------------------------- #
exp = "exp004"
config = Config(EXP_NAME=exp, TARGET="CLASS")
exp_env.make_env(config)
# ---------------------------------------------------------------------- #


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


def get_function(block, is_train):
    s = mapping = {
        True: 'fit',
        False: 'transform'
    }.get(is_train)
    return getattr(block, s)


def to_feature(input_df,
               blocks,
               is_train=False):
    out_df = pd.DataFrame()

    for block in tqdm(blocks, total=len(blocks)):
        func = get_function(block, is_train)

        with timer(prefix='create ' + str(block) + ' '):
            _df = func(input_df)
        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)
    return reduce_mem_usage(out_df)


# make KFold
def make_kf(train_x, train_y, n_splits, random_state=71):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    s = 5
    # _y = pd.cut(train_y, s, labels=range(s))
    return list(kf.split(train_x, train_y))


def make_skf(train_x, train_y, n_splits, random_state=71):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(train_x, train_y))


# plot result
def result_plot(train_y, oof):
    name = "result"
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.distplot(train_y, label='train_y', color='orange')
    sns.distplot(oof, label='oof')
    ax.legend()
    ax.grid()
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(config.REPORTS, f'{name}.png'), dpi=120)  # save figure
    plt.show()


def result_confusion_matrix(y_true,
                            pred_label,
                            ax: Optional[plt.Axes] = None,
                            labels: Optional[list] = None,
                            conf_options: Optional[dict] = None,
                            plot_options: Optional[dict] = None) -> Tuple[plt.Axes, np.array]:

    _conf_options = {
        "normalize": "true",
    }
    if conf_options is not None:
        _conf_options.update(conf_options)

    _plot_options = {
        "cmap": "Blues",
        "annot": True
    }
    if plot_options is not None:
        _plot_options.update(plot_options)

    conf = confusion_matrix(y_true=y_true,
                            y_pred=pred_label,
                            **_conf_options)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf, ax=ax, **_plot_options)
    ax.set_ylabel("Label")
    ax.set_xlabel("Predict")

    if labels is not None:
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        ax.tick_params("y", labelrotation=0)
        ax.tick_params("x", labelrotation=90)

    name = "confusion"
    fig.tight_layout()
    fig.savefig(os.path.join(config.REPORTS, f'{name}.png'), dpi=120)  # save figure
    plt.show()


# create submission
def create_submission(preds):
    sample_sub = pd.read_csv(os.path.join(config.INPUT, "sample_submission.csv"))
    # post_preds = [0 if x < 0 else x for x in preds]
    sample_sub.iloc[:, 1:] = preds
    sample_sub.to_csv(os.path.join(config.SUBMISSION, f'{config.EXP_NAME}.csv'), index=False)


def preprocess(input_df):
    """
    input: train, test
    """

    # ジャンルは文字列に変更する
    output_df = input_df.copy()

    output_df["biggenre"] = output_df["biggenre"].astype(str)
    output_df["genre"] = output_df["genre"].astype(str)

    return output_df


def processing_ncode(input_df: pd.DataFrame):
    output_df = input_df.copy()

    num_dict = {chr(i): i - 65 for i in range(65, 91)}

    def _processing(x, num_dict=num_dict):
        y = 0
        for i, c in enumerate(x[::-1]):
            num = num_dict[c]
            y += 26 ** i * num
        y *= 9999
        return y

    tmp_df = pd.DataFrame()
    tmp_df['_ncode_num'] = input_df['ncode'].map(lambda x: x[1:5]).astype(int)
    tmp_df['_ncode_chr'] = input_df['ncode'].map(lambda x: x[5:])
    tmp_df['_ncode_chr2num'] = tmp_df['_ncode_chr'].map(lambda x: _processing(x))

    output_df['ncode_num'] = tmp_df['_ncode_num'] + tmp_df['_ncode_chr2num']
    return output_df['ncode_num']


def get_past_days(input_df: pd.DataFrame):
    output_df = pd.DataFrame()
    dt_now = datetime.datetime(2021, 9, 24)  # コンペ開催時
    output_df['past_days'] = input_df['general_firstup']\
        .apply(lambda x: (dt_now - datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)

    return output_df


# StringLenBlock
class StringLengthBlock(BaseBlock):
    def __init__(self, column):
        self.column = column

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.column] = input_df[self.column].str.len()
        return output_df.add_prefix('StringLength_')


# umapで次元圧縮している
def get_umap_title(input_df):
    col_prefix = "title"
    if len(input_df) == 40000:
        titles = np.load("../add_data/train_title_roberta.npy")
    else:
        titles = np.load("../add_data/test_title_roberta.npy")

    title_df = pd.DataFrame(titles)
    um = umap.UMAP(n_components=100, random_state=71)
    _umap = um.fit_transform(title_df)
    umap_df = pd.DataFrame(_umap, columns=[f'{col_prefix}_umap{c}' for c in range(1, 101)])
    return umap_df


def get_umap_story(input_df):
    col_prefix = "story"
    if len(input_df) == 40000:
        stories = np.load("../add_data/train_story_roberta.npy")
    else:
        stories = np.load("../add_data/test_story_roberta.npy")

    story_df = pd.DataFrame(stories)
    um = umap.UMAP(n_components=100, random_state=71)
    _umap = um.fit_transform(story_df)
    umap_df = pd.DataFrame(_umap, columns=[f'{col_prefix}_umap{c}' for c in range(1, 101)])

    return umap_df


def get_umap_keyword(input_df):
    col_prefix = "keyword"
    if len(input_df) == 40000:
        keywords = np.load("../add_data/train_keyword_roberta.npy")
    else:
        keywords = np.load("../add_data/test_keyword_roberta.npy")

    keyword_df = pd.DataFrame(keywords)
    um = umap.UMAP(n_components=100, random_state=71)
    _umap = um.fit_transform(keyword_df)
    umap_df = pd.DataFrame(_umap, columns=[f'{col_prefix}_umap{c}' for c in range(1, 101)])
    return umap_df


def get_shosekika(x):
    if "書籍化" in x:
        return 1
    else:
        return 0


def get_comicalize(x):
    if "コミカライズ" in x:
        return 1
    else:
        return 0


def get_flag_s_c(input_df):
    output_df = pd.DataFrame()

    output_df["isshosekika"] = input_df["story"].apply(lambda x: get_shosekika(x))
    output_df["iscomicalize"] = input_df["story"].apply(lambda x: get_comicalize(x))

    return output_df


# ---------------------------------------------------------------------- #
def main():
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    train_df = pd.read_csv(os.path.join(config.INPUT, "train.csv"))
    test_df = pd.read_csv(os.path.join(config.INPUT, "test.csv"))

    train = preprocess(train_df)
    test = preprocess(test_df)

    whole_df = pd.concat([train, test], axis=0)

    process_blocks = [
        *[ContinuousBlock(c) for c in [
            "novel_type",
            "isr15",
            "isbl",
            "isgl",
            "iszankoku",
            "istensei",
            "istenni",
            "pc_or_k"
        ]],
        *[CountEncodingBlock(c, whole_df=whole_df) for c in [
            "biggenre",
            "genre",
            "writer"
        ]],
        *[LabelEncodingBlock(c, whole_df=whole_df) for c in [
            "biggenre",
            "genre",
            "writer"
        ]],
        WrapperBlock(processing_ncode),
        WrapperBlock(get_past_days),
        *[StringLengthBlock(c) for c in [
            "title",
            "story"
        ]],
        WrapperBlock(get_umap_title),
        WrapperBlock(get_umap_story),
        WrapperBlock(get_umap_keyword),
        WrapperBlock(get_flag_s_c)
    ]

    # create train_x, train_y, test_x
    train_y = train["fav_novel_cnt_bin"]
    train_x = to_feature(train, process_blocks, is_train=True)
    test_x = to_feature(test, process_blocks)

    # dump features
    joblib.dump(train_x, os.path.join("../output/" + exp + "/feature", "train_feat.pkl"))
    joblib.dump(test_x, os.path.join("../output/" + exp + "/feature", "test_feat.pkl"))

    # set model
    model = MyLGBMClassifier

    # set run params
    run_params = {
        "metrics": log_loss,
        "cv": make_skf,
        "feature_select_method": "tree_importance",
        "feature_select_fold": 5,
        "feature_select_num": 500,
        "folds": 5,
        "seeds": [71, 72, 73],
    }

    # set model params
    model_params = {
        "n_estimators": 20000,
        "objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "learning_rate": 0.01,
        "num_leaves": 256,
        "n_jobs": -1,
        "importance_type": "gain",
        "reg_lambda": .7,
        "colsample_bytree": 1,
        "max_depth": 7,
    }

    # fit params
    fit_params = {
        "early_stopping_rounds": 100,
        "verbose": 1000
    }

    # features
    features = {
        "train_x": train_x,
        "test_x": test_x,
        "train_y": train_y
    }

    # run model
    config.RUN_NAME = f'_{config.TARGET}'
    runner = Runner(config=config,
                    run_params=run_params,
                    model_params=model_params,
                    fit_params=fit_params,
                    model=model,
                    features=features,
                    use_mlflow=False
                    )
    runner.run_train_cv()
    runner.run_predict_cv()

    # make_submission
    create_submission(preds=runner.preds)

    # plot result
    # result_plot(train_y=train_y, oof=runner.oof)
    result_confusion_matrix(y_true=train_y,
                            pred_label=np.argmax(runner.oof, axis=1),
                            conf_options={"normalize": None},
                            plot_options={"fmt": "4d"})


if __name__ == "__main__":
    main()