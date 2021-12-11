
import os
import warnings
import sys
import joblib
import re
import regex

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
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from gensim.models import word2vec, KeyedVectors

import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency

sys.path.append("../")

from mypipe.config import Config
from mypipe.utils import Util
from mypipe.utils import reduce_mem_usage, seed_everything
from mypipe.experiment import exp_env
from mypipe.experiment.runner import Runner
from mypipe.models.model_lgbm import MyLGBMClassifier
from mypipe.Block_features import BaseBlock, ContinuousBlock, CountEncodingBlock, OneHotEncodingBlock, \
    LabelEncodingBlock, ArithmeticOperationBlock, AggregationBlock, WrapperBlock


# ---------------------------------------------------------------------- #
exp = "exp038"
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

    # 日付データの作成
    output_df["general_firstup2"] = pd.to_datetime(output_df["general_firstup"])
    output_df["Year"] = output_df["general_firstup2"].dt.year
    output_df["Month"] = output_df["general_firstup2"].dt.month
    output_df["Day"] = output_df["general_firstup2"].dt.day
    output_df["publish_hour"] = output_df["general_firstup2"].dt.hour
    output_df["publish_min"] = output_df["general_firstup2"].dt.minute
    output_df["publish_weeknum"] = output_df["general_firstup2"].dt.dayofweek

    # Ndode変換
    output_df["ncode_num"] = processing_ncode(input_df)

    # 経過日数入れ込み
    output_df["past_days"] = get_past_days(output_df)

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

    return output_df['past_days']


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


def get_award(x):
    if "大賞" in x:
        return 1
    else:
        return 0


def get_flag_s_c(input_df):
    """書籍化とかコミカライズ、大賞受賞しているものは人気ではないか"""
    output_df = pd.DataFrame()
    # どこの欄にいても良いように変更

    output_df["isshosekika_t"] = input_df["title"].apply(lambda x: get_shosekika(x))
    output_df["isshosekika_s"] = input_df["story"].apply(lambda x: get_shosekika(x))
    output_df["isshosekika_k"] = input_df["keyword"].fillna("nan").apply(lambda x: get_shosekika(x))

    output_df["isshosekika"] = output_df[["isshosekika_t", "isshosekika_s", "isshosekika_k"]].max(axis=1)

    output_df["iscomicalize_t"] = input_df["title"].apply(lambda x: get_comicalize(x))
    output_df["iscomicalize_s"] = input_df["story"].apply(lambda x: get_comicalize(x))
    output_df["iscomicalize_k"] = input_df["keyword"].fillna("nan").apply(lambda x: get_comicalize(x))

    output_df["iscomicalize"] = output_df[["iscomicalize_t", "iscomicalize_s", "iscomicalize_k"]].max(axis=1)

    output_df["isaward_t"] = input_df["title"].apply(lambda x: get_award(x))
    output_df["isaward_s"] = input_df["story"].apply(lambda x: get_award(x))
    output_df["isaward_k"] = input_df["keyword"].fillna("nan").apply(lambda x: get_award(x))

    output_df["isaward"] = output_df[["isaward_t", "isaward_s", "isaward_k"]].max(axis=1)

    return output_df[["isaward", "isshosekika", "iscomicalize"]]


def get_mrmmo(x):
    if "VRMMO" in x:
        return 1
    else:
        return 0


def get_flag_saikin(input_df):
    """最近流行りのキーワードをフラグで立てる"""
    output_df = pd.DataFrame()

    output_df["isvrmmo"] = input_df["keyword"].fillna("").apply(lambda x: 1 if "VRMMO" in x else 0)
    output_df["isakuyakureijou"] = input_df["keyword"].fillna("").apply(lambda x: 1 if "悪役令嬢" in x else 0)
    output_df["isakuyaku"] = input_df["title"].fillna("").apply(lambda x: 1 if "悪役" in x else 0)
    output_df["isvrmmo"] = input_df["keyword"].fillna("").apply(lambda x: 1 if "異世界転生" in x else 0)
    output_df["issaikyo"] = input_df["keyword"].fillna("").apply(lambda x: 1 if "主人公最強" in x else 0)
    output_df["isonnasyujinko"] = input_df["keyword"].fillna("").apply(lambda x: 1 if "女主人公" in x else 0)
    output_df["isknight"] = input_df["keyword"].fillna("").apply(lambda x: 1 if "騎士" in x else 0)
    output_df["isgakuen"] = input_df["keyword"].fillna("").apply(lambda x: 1 if "学園" in x else 0)
    output_df["isotomegame"] = input_df["keyword"].fillna("").apply(lambda x: 1 if "乙女ゲーム" in x else 0)

    return output_df


def count_keyword(input_df):
    """キーワードの数を数える"""
    output_df = pd.DataFrame()
    output_df["keyword_count"] = input_df["keyword"].fillna("nan").str.split(" ").apply(len)

    return output_df


def mecab_tokenizer(s: str):
    tagger = MeCab.Tagger("mecabrc")
    parse_result = tagger.parse(s)
    return [
        result.split("\t")[0]
        for result in parse_result.split("\n")
        if result not in ["EOS", ""]
    ]


class TfidfBlock(BaseBlock):
    def __init__(self, column: str, decomposition: str = "svd"):
        self.column = column
        self.decomposition = decomposition


    def get_master(self, input_df):
        """tdidfを計算するための全体集合を返す.
        デフォルトでは fit でわたされた dataframe を使うが, もっと別のデータを使うのも考えられる."""
        return input_df

    def fit(self, input_df, y=None):
        master_df = self.get_master(input_df)
        text = input_df[self.column].fillna("")

        if self.decomposition == "svd":
            self.pipeline_ = Pipeline([
                ("tfidf", TfidfVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("svd", TruncatedSVD(n_components=50, random_state=71)),
            ])

        elif self.decomposition == "NMF":
            self.pipeline_ = Pipeline([
                ("tfidf", TfidfVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("NMF", NMF(n_components=50, random_state=71))
            ])

        elif self.decomposition == "LDA":
            self.pipeline_ = Pipeline([
                ("tfidf", TfidfVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("LDA", LatentDirichletAllocation(n_components=50, random_state=71))
            ])

        else:
            self.pipeline_ = Pipeline([
                ("tfidf", TfidfVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("svd", TruncatedSVD(n_components=50, random_state=71)),
            ])

        self.pipeline_.fit(text)

        return self.transform(input_df)

    def transform(self, input_df):
        text = input_df[self.column].fillna("")
        z = self.pipeline_.transform(text)

        out_df = pd.DataFrame(z)
        return out_df.add_prefix(f'{self.column}_tfidf_{self.decomposition}_')


class RankBlock(BaseBlock):
    def __init__(self, whole_df: pd.DataFrame,
                 column: str,
                 key: str,
                 rank_method: str = "average",
                 ascending=False):
        self.whole_df = whole_df
        self.column = column
        self.key = key
        self.rank_method = rank_method
        self.ascending = ascending

    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        self.rank_df_ = self.whole_df[[self.key, self.column]].groupby(self.key).rank(method=self.rank_method,
                                                                                     ascending=self.ascending)
        self.rank_df = pd.concat([self.whole_df["ncode"], self.rank_df_], axis=1)
        column_name = [f'RANK_{self.key}@{self.column}']
        self.rank_df.columns = ["ncode"] + column_name

        output_df = pd.merge(input_df, self.rank_df, on="ncode", how="left")

        return output_df[column_name]


# reference: https://github.com/arosh/BM25Transformer/blob/master/bm25.py
# https://zenn.dev/koukyo1994/articles/9b1da2482d8ba1
class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # Average document length
        # Scalar value
        avgdl = np.average(dl)
        # Compute BM25 score only for non-zero elements
        data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        return X


class BM25Block(BaseBlock):
    def __init__(self, column: str, decomposition: str = "svd"):
        self.column = column
        self.decomposition = decomposition

    def get_master(self, input_df):
        """tdidfを計算するための全体集合を返す.
        デフォルトでは fit でわたされた dataframe を使うが, もっと別のデータを使うのも考えられる."""
        return input_df

    def fit(self, input_df, y=None):
        master_df = self.get_master(input_df)
        text = input_df[self.column].fillna("")

        if self.decomposition == "svd":
            self.pipeline_ = Pipeline([
                ("CountVectorizer", CountVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("BM25Transformer", BM25Transformer()),
                ("svd", TruncatedSVD(n_components=50, random_state=71)),
            ])

        elif self.decomposition == "NMF":
            self.pipeline_ = Pipeline([
                ("CountVectorizer", CountVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("BM25Transformer", BM25Transformer()),
                ("NMF", NMF(n_components=50, random_state=71))
            ])

        elif self.decomposition == "LDA":
            self.pipeline_ = Pipeline([
                ("CountVectorizer", CountVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("BM25Transformer", BM25Transformer()),
                ("LDA", LatentDirichletAllocation(n_components=50, random_state=71))
            ])

        else:
            self.pipeline_ = Pipeline([
                ("CountVectorizer", CountVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("BM25Transformer", BM25Transformer()),
                ("svd", TruncatedSVD(n_components=50, random_state=71)),
            ])

        self.pipeline_.fit(text)

        return self.transform(input_df)

    def transform(self, input_df):
        text = input_df[self.column].fillna("")
        z = self.pipeline_.transform(text)

        out_df = pd.DataFrame(z)
        return out_df.add_prefix(f'{self.column}_BM25_{self.decomposition}_')


def monthly_publish(input_df):
    monthly_df = input_df.groupby(["writer", "Year", "Month"]).count()["ncode"].reset_index()
    monthly_df.rename(columns={"ncode": "monthly_publish"}, inplace=True)

    output_df = pd.merge(input_df, monthly_df, on=["writer", "Year", "Month"], how="left")

    return output_df["monthly_publish"]


def count_kaigyo(x):
  return x.count("\n")


def get_k(input_df):
    output_df = pd.DataFrame()
    output_df["kaigyos"] = input_df["story"].apply(lambda x: count_kaigyo(x))
    return output_df


def create_type_features(texts):
    """文字種ベースの特徴量を作成"""

    type_data = []

    for text in texts:
        tmp = []

        tmp.append(len(text))

        # 平仮名の文字数カウント
        p = re.compile('[\u3041-\u309F]+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        # カタカナの文字数カウント
        p = re.compile('[\u30A1-\u30FF]+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        # 漢字の文字数カウント
        p = regex.compile(r'\p{Script=Han}+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        # 絵文字の文字数カウント
        p = regex.compile(r'\p{Emoji_Presentation=Yes}+')
        s = ''.join(p.findall(text))
        tmp.append(len(s))

        type_data.append(tmp)

    colnames = ['length', 'hiragana_length', 'katakana_length', 'kanji_length', 'emoji_length']
    type_df = pd.DataFrame(type_data, columns=colnames)

    for colname in type_df.columns:
        if colname != 'length':
            type_df[colname] /= type_df['length']

    return type_df


def get_letter_ratio1(input_df):
    output_df = create_type_features(input_df["title"].values)
    return output_df.add_prefix("title_")


def get_letter_ratio2(input_df):
    output_df = create_type_features(input_df["story"].values)
    # 空白数を追加
    output_df["blank_count"] = input_df["story"].apply(lambda x: x.count('\u3000'))
    output_df["blank_ratio"] = output_df["blank_count"] / output_df["length"]
    return output_df.add_prefix("story_")


class GroupDiffBlock(BaseBlock):
    def __init__(self, whole_df: pd.DataFrame, key: str, column: str,  diff: int):
        self.whole_df = whole_df
        self.key = key
        self.column = column
        self.diff = diff

    def transform(self, input_df):
        gp_diff_df = self.whole_df.groupby(self.key)[self.column].diff(self.diff).to_frame()
        gp_diff_df.rename(columns={self.column: f'Diff_{self.key}_{self.column}_{self.diff}'}, inplace=True)
        gp_df = pd.concat([self.whole_df[["ncode"]].reset_index(drop=True), gp_diff_df.reset_index(drop=True)], axis=1)

        output_df = pd.merge(input_df, gp_df, on="ncode", how="left")

        return output_df[f'Diff_{self.key}_{self.column}_{self.diff}']


class CountVectorizerBlock(BaseBlock):
    def __init__(self, column: str, decomposition: str = "svd"):
        self.column = column
        self.decomposition = decomposition

    def get_master(self, input_df):
        """tdidfを計算するための全体集合を返す.
        デフォルトでは fit でわたされた dataframe を使うが, もっと別のデータを使うのも考えられる."""
        return input_df

    def fit(self, input_df, y=None):
        master_df = self.get_master(input_df)
        text = input_df[self.column].fillna("")

        if self.decomposition == "svd":
            self.pipeline_ = Pipeline([
                ("CountVectorizer", CountVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("svd", TruncatedSVD(n_components=50, random_state=71)),
            ])

        elif self.decomposition == "NMF":
            self.pipeline_ = Pipeline([
                ("CountVectorizer", CountVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("NMF", NMF(n_components=50, random_state=71))
            ])

        elif self.decomposition == "LDA":
            self.pipeline_ = Pipeline([
                ("CountVectorizer", CountVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("LDA", LatentDirichletAllocation(n_components=50, random_state=71))
            ])

        else:
            self.pipeline_ = Pipeline([
                ("CountVectorizer", CountVectorizer(tokenizer=mecab_tokenizer, max_features=10000)),
                ("svd", TruncatedSVD(n_components=50, random_state=71)),
            ])

        self.pipeline_.fit(text)

        return self.transform(input_df)

    def transform(self, input_df):
        text = input_df[self.column].fillna("")
        z = self.pipeline_.transform(text)

        out_df = pd.DataFrame(z)
        return out_df.add_prefix(f'{self.column}_CountVectorizer_{self.decomposition}_')


class KeywordOneHotEncodingBlock(BaseBlock):
    def __init__(self, whole_df: pd.DataFrame, limit: int):
        self.whole_df = whole_df
        self.limit = limit

    def transform(self, input_df):
        output_df = pd.DataFrame()
        keyword_list = self.whole_df["keyword"].fillna("").str.split(" ").explode().value_counts().head(self.limit).index.to_list()
        for keyword in keyword_list:
            output_df[f'is_{keyword}'] = input_df["keyword"].fillna("").apply(lambda x: 1 if keyword in x else 0)

        return output_df


class ShiftGenreRatioBlock(BaseBlock):
    """その月・年の前の情報（投稿数）を特徴量にする"""
    def __init__(self, column: str, months: bool, shifts: int):
        self.column = column
        self.months = months
        self.shifts = shifts

    def fit(self, input_df, y=None):
        if self.months:
            cross_df = pd.crosstab([input_df["Year"], input_df["Month"]],
                                   input_df[self.column],
                                    normalize="index").reset_index()

            self.shift_df = cross_df[["Year", "Month"]]

            for i in cross_df.columns:
                if i == "Year":
                    pass
                elif i == "Month":
                    pass
                else:
                    self.shift_df[f"Shift_{self.column}_Month_ratio_{self.shifts}"] = cross_df[i].shift(self.shifts)
        else:
            cross_df = pd.crosstab([input_df["Year"]],
                                   input_df[self.column],
                                   normalize="index").reset_index()

            self.shift_df = cross_df[["Year"]]

            for i in cross_df.columns:
                if i == "Year":
                    pass
                else:
                    self.shift_df[f"Shift_{self.column}_ratio_{self.shifts}"] = cross_df[i].shift(self.shifts)

        return self.transform(input_df)

    def transform(self, input_df):
        if self.months:
            output_df = pd.merge(input_df[["Year", "Month"]], self.shift_df, on=["Year", "Month"], how="left")
            output_df = output_df.drop(["Year", "Month"], axis=1)

        else:
            output_df = pd.merge(input_df[["Year"]], self.shift_df, on="Year", how="left")
            output_df = output_df.drop("Year", axis=1)

        return output_df


class LagCumBlock(BaseBlock):
    def __init__(self, whole_df, column, lag):
        self.whole_df = whole_df
        self.column = column
        self.lag = lag

    def transform(self, input_df):
        gp_total = self.whole_df.groupby(pd.Grouper(key="general_firstup2", freq="D")).count()["ncode"].reset_index()
        gp_total["Year"] = gp_total["general_firstup2"].dt.year
        gp_total["Month"] = gp_total["general_firstup2"].dt.month
        gp_total["Day"] = gp_total["general_firstup2"].dt.day
        gp_total = gp_total.drop("general_firstup2", axis=1)
        gp_total.rename(columns={"ncode": "daily_novels"}, inplace=True)

        gp_df2 = pd.crosstab([self.whole_df["Year"],
                              self.whole_df["Month"],
                              self.whole_df["Day"]],
                             self.whole_df[self.column]).reset_index()

        gp_total = pd.merge(gp_total, gp_df2, on=["Year", "Month", "Day"], how="left")

        _out = gp_total[["Year", "Month", "Day"]]

        for i in gp_total.columns:
            if i == "Year":
                pass
            elif i == "Month":
                pass
            elif i == "Day":
                pass
            else:
                _out[f"Lag_genre_{i}_{self.lag}"] = gp_total[i].shift(self.lag)

        output_df = pd.merge(input_df[["Year", "Month", "Day"]], _out, on=["Year", "Month", "Day"], how="left")
        output_df = output_df.drop(["Year", "Month", "Day"], axis=1)

        return output_df


def get_writer_year(x):
    if x < 400:
        return 2004
    elif x < 2000:
        return 2005
    elif x < 7500:
        return 2006
    elif x < 19000:
        return 2007
    elif x < 32000:
        return 2008
    elif x < 59000:
        return 2009
    elif x < 120000:
        return 2010
    elif x < 200000:
        return 2011
    elif x < 300000:
        return 2012
    elif x < 400000:
        return 2013
    elif x < 550000:
        return 2014
    elif x < 730000:
        return 2015
    elif x < 960000:
        return 2016
    elif x < 1240000:
        return 2017
    else:
        return 2018


def get_creator_year(input_df):
    output_df = pd.DataFrame()
    output_df["writer_years"] = input_df["userid"].apply(lambda x: get_writer_year(x))
    return output_df


class ShiftMeanFavBlock(BaseBlock):
    def __init__(self, whole_df, column):
        self.whole_df = whole_df
        self.column = column

    def fit(self, input_df, y=None):
        # 年月のdfを作成する
        gp_months = self.whole_df.groupby(pd.Grouper(key="general_firstup2", freq="M")).mean()["userid"].reset_index()
        gp_months["Year"] = gp_months["general_firstup2"].dt.year
        gp_months["Month"] = gp_months["general_firstup2"].dt.month
        gp_months = gp_months.drop(["general_firstup2", "userid"], axis=1)

        # 月単位の平均bookmarksを算出してshiftをし（2個）する
        mean_dfs = []
        for i in input_df[self.column].unique():
            mean_df = input_df[input_df[self.column]==i].groupby([self.column,
                                                                  "Year",
                                                                  "Month"]).mean()["fav_novel_cnt_bin"].shift(2).reset_index()
            mean_df.rename(columns={"fav_novel_cnt_bin": "mean_novel_fav"}, inplace=True)
            mean_dfs.append(mean_df)
        mean_df = pd.concat(mean_dfs)

        self.month_mean_fav_df = pd.merge(gp_months, mean_df, on=["Year", "Month"], how="left").fillna(0)

        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.merge(input_df[[self.column, "Year", "Month"]],
                             self.month_mean_fav_df,
                             on=[self.column, "Year", "Month"],
                             how="left")

        return output_df["mean_novel_fav"]


# ジャスト投稿かどうかのフラグ
def just_flag(x):
    if x[-5:] == "00:00":
        return 1
    else:
        return 0

def get_just(input_df):
    output_df = pd.DataFrame()
    output_df["is_just_time"] = input_df["general_firstup"].apply(lambda x: just_flag(x))

    return output_df


# キーワードをword2vecで変換する
class KeyWord2VecBlock(BaseBlock):
    def __init__(self, whole_df):
        self.whole_df = whole_df

    def fit(self, input_df, y=None):
        _temp = self.whole_df[["ncode", "keyword"]].copy().reset_index(drop=True)
        _temp["keyword"] = _temp["keyword"].str.split(' ')
        w2v_model = word2vec.Word2Vec(_temp["keyword"].fillna("nan").values.tolist(),
                                      size=10,
                                      min_count=1,
                                      window=1,
                                      iter=100)
        sentence_vectors = _temp["keyword"].fillna("nan").apply(lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0))
        sentence_vectors = np.vstack([x for x in sentence_vectors])
        self.sentence_vector_df = pd.DataFrame(sentence_vectors, columns=[f'keyword_w2v_{i}' for i in range(10)])
        self.sentence_vector_df["ncode"] = _temp["ncode"]

        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.merge(input_df, self.sentence_vector_df, on="ncode", how="left")

        return output_df[[f'keyword_w2v_{i}' for i in range(10)]]


# ---------------------------------------------------------------------- #
def main():
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    seed_everything(71)
    
    train_df = pd.read_csv(os.path.join(config.INPUT, "train_2021-2020.csv"))
    test_df = pd.read_csv(os.path.join(config.INPUT, "test.csv"))

    train = preprocess(train_df)
    test = preprocess(test_df)
    # ラベルを2値分類した結果を追加（リークになるか？）
    label_34_df_tr = pd.read_csv("../add_data/pred_train_label_3_4.csv")
    label_34_df_ts = pd.read_csv("../add_data/pred_test_label_3_4.csv")

    label_3_df_tr = pd.read_csv("../add_data/pred_train_label_3.csv")
    label_3_df_ts = pd.read_csv("../add_data/pred_test_label_3.csv")

    label_2_df_tr = pd.read_csv("../add_data/pred_train_label_2.csv")
    label_2_df_ts = pd.read_csv("../add_data/pred_test_label_2.csv")

    label_1_df_tr = pd.read_csv("../add_data/pred_train_label_1.csv")
    label_1_df_ts = pd.read_csv("../add_data/pred_test_label_1.csv")

    label_0_df_tr = pd.read_csv("../add_data/pred_train_label_0.csv")
    label_0_df_ts = pd.read_csv("../add_data/pred_test_label_0.csv")

    train = pd.merge(train, label_34_df_tr, on="ncode", how="left")
    test = pd.merge(test, label_34_df_ts, on="ncode", how="left")

    train = pd.merge(train, label_3_df_tr, on="ncode", how="left")
    test = pd.merge(test, label_3_df_ts, on="ncode", how="left")

    train = pd.merge(train, label_2_df_tr, on="ncode", how="left")
    test = pd.merge(test, label_2_df_ts, on="ncode", how="left")
    train = pd.merge(train, label_1_df_tr, on="ncode", how="left")
    test = pd.merge(test, label_1_df_ts, on="ncode", how="left")

    train = pd.merge(train, label_0_df_tr, on="ncode", how="left")
    test = pd.merge(test, label_0_df_ts, on="ncode", how="left")

    # 2021~2019のデータでの予測結果を追加する
    # ncode結合できてないのでおかしいのでは？
    train_2019 = pd.read_csv(os.path.join(config.INPUT, "train_2021-2019.csv"))

    _oof = Util.load("../output/exp030/preds/oof_CLASS.pkl")
    _oof_df = pd.DataFrame(_oof)

    _pred = Util.load("../output/exp030/preds/preds_CLASS.pkl")
    _pred_df = pd.DataFrame(_pred)

    oof_columns = [f"pred_2019_{i:02d}" for i in range(5)]
    _oof_df.columns = oof_columns
    _pred_df.columns = oof_columns

    # 変更
    _oof_df["ncode"] = train_2019["ncode"]

    train = pd.merge(train, _oof_df, on="ncode", how="left")
    test = test.join(_pred_df)

    # 全データの予測結果を追加する
    train_all = pd.read_csv(os.path.join(config.INPUT, "train.csv"))
    _oof_all = Util.load("../output/exp028/preds/oof_CLASS.pkl")
    _oof_all_df = pd.DataFrame(_oof_all)

    _pred_all = Util.load("../output/exp028/preds/preds_CLASS.pkl")
    _pred_all_df = pd.DataFrame(_pred_all)

    oof_columns2 = [f"pred_all_{i:02d}" for i in range(5)]
    _oof_all_df.columns = oof_columns2
    _pred_all_df.columns = oof_columns2

    # 変更
    _oof_all_df["ncode"] = train_all["ncode"]

    train = pd.merge(train, _oof_all_df, on="ncode", how="left")
    test = test.join(_pred_all_df)

    # 結合データ作成
    whole_df = pd.concat([train, test], axis=0)

    process_blocks = [
        KeyWord2VecBlock(whole_df=whole_df),
        ShiftMeanFavBlock(whole_df=whole_df, column="genre"),
        WrapperBlock(get_creator_year),
        WrapperBlock(get_just),
        *[LagCumBlock(whole_df=whole_df, column="genre", lag=c) for c in [1, 2, 3, 4, 5, 6, 7]],
        GroupDiffBlock(whole_df=whole_df, key="writer", column="past_days", diff=1),
        *[ShiftGenreRatioBlock(column="genre", months=True, shifts=c) for c in [1, 2, 3]],
        *[ShiftGenreRatioBlock(column="genre", months=False, shifts=c) for c in [1, 2, 3]],
        *[ShiftGenreRatioBlock(column="biggenre", months=True, shifts=c) for c in [1, 2, 3]],
        *[ShiftGenreRatioBlock(column="biggenre", months=False, shifts=c) for c in [1, 2, 3]],
        KeywordOneHotEncodingBlock(whole_df=whole_df, limit=500),
        *[ContinuousBlock(c) for c in [
            "novel_type",
            "isr15",
            "isbl",
            "isgl",
            "iszankoku",
            "istensei",
            "istenni",
            "pc_or_k",
            "Year",
            "Month",
            "Day",
            "ncode_num",
            "past_days",
            "publish_hour",
            "publish_min",
            "publish_weeknum",
            "pred_label_3_4",
            "pred_label_3",
            "pred_label_2",
            "pred_label_1",
            "pred_label_0",
            "pred_2019_00",
            "pred_2019_01",
            "pred_2019_02",
            "pred_2019_03",
            "pred_2019_04",
            "pred_all_00",
            "pred_all_01",
            "pred_all_02",
            "pred_all_03",
            "pred_all_04"
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
        *[OneHotEncodingBlock(c, count_limit=50) for c in [
            "writer",
            "genre"
        ]],
        *[StringLengthBlock(c) for c in [
            "title",
            "story",
            "keyword"
        ]],
        WrapperBlock(get_flag_s_c),
        WrapperBlock(count_keyword),
        # WrapperBlock(get_flag_saikin),
        *[TfidfBlock(c, decomposition="svd") for c in[
            "keyword",
            "title",
            "story"
        ]],
        *[TfidfBlock(c, decomposition="NMF") for c in [
            "keyword",
            "title",
            "story"
        ]],
        *[TfidfBlock(c, decomposition="LDA") for c in [
            "keyword",
            "title",
            "story"
        ]],
        *[RankBlock(whole_df=whole_df, key=c, column="ncode_num") for c in [
            "writer",
        ]],
        *[BM25Block(c, decomposition="NMF") for c in [
            "keyword",
            "title",
        ]],
        *[BM25Block(c, decomposition="svd") for c in [
            "keyword",
            "title",
        ]],
        *[BM25Block(c, decomposition="LDA") for c in [
            "keyword",
            "title",
        ]],
        WrapperBlock(monthly_publish),
        WrapperBlock(get_k),
        WrapperBlock(get_letter_ratio1),
        WrapperBlock(get_letter_ratio2),
        *[AggregationBlock(whole_df=whole_df,
                           key="writer",
                           agg_column=c,
                           agg_funcs=["mean", "max", "min", "std"]) for c in [
            "publish_hour",
            "Month",
            "past_days",
            "pred_label_3_4",
            "pred_label_3",
            "pred_label_2",
            "pred_label_1",
            "pred_label_0",
            "pred_2019_00",
            "pred_2019_01",
            "pred_2019_02",
            "pred_2019_03",
            "pred_2019_04",
            "pred_all_00",
            "pred_all_01",
            "pred_all_02",
            "pred_all_03",
            "pred_all_04"
        ]],
        *[AggregationBlock(whole_df=whole_df,
                           key="genre",
                           agg_column=c,
                           agg_funcs=["mean", "max", "min", "std"]) for c in [
            "publish_hour",
            "Month",
            "past_days",
            "pred_label_3_4",
            "pred_label_3",
            "pred_label_2",
            "pred_label_1",
            "pred_label_0",
            "pred_2019_00",
            "pred_2019_01",
            "pred_2019_02",
            "pred_2019_03",
            "pred_2019_04",
            "pred_all_00",
            "pred_all_01",
            "pred_all_02",
            "pred_all_03",
            "pred_all_04"
          ]],
        *[CountVectorizerBlock(c, decomposition="svd") for c in [
            "keyword",
            "title",
            "story"
        ]],
        *[CountVectorizerBlock(c, decomposition="NMF") for c in [
            "keyword",
            "title",
            "story"
        ]],
        *[CountVectorizerBlock(c, decomposition="LDA") for c in [
            "keyword",
            "title",
            "story"
        ]],
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
        "feature_select_num": 3500,
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
        "num_leaves": 1023,
        "n_jobs": -1,
        "importance_type": "gain",
        "reg_lambda": .7,
        "colsample_bytree": 0.8,
        "colsample_bynode": 0.6,
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