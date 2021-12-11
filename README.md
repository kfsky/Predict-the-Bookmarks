# Predict-the-Bookmarks
<B>小説家になろう ブクマ数予測 ”伸びる”タイトルとは？</B>  
日本最大級の小説投稿サイトである“小説家になろう”様のデータを用いて、
ジャンルや作者名などの関連データから各小説のブックマーク数（５段階にBIn化）を予測する。  
➟多クラス分類

## Summary
Public 20位 Private 19位で終了。

## Solution
以下のような形でFinish。基本的にLightGBMでの実装がメインで行っていった。
最終的には３モデルの予測値を特徴量にしたLightGBMのモデルがprivateで最も精度が良かった。
（スタッキングが効果あったかも知れない）

## Feature Engineer
基本的な特徴量作成を行いながら、自然言語部分の処理を行っていった。
### 基本特徴量
* OneHotEncoding
* LabelEncoding
* CountEncoding
* TargetEncoding
* StringLength（タイトルなどの文字の長さ）
### 自然言語処理
* TfIdfBlock（Tfidfを行い、svd, LDA, NMFそれぞれで50次元まで削減した特徴量）
* BM25Block（BM25を行い、svd, LDA, NMFそれぞれで50次元まで削減した特徴量）
* CountVectorizerBlock（CountVectorizerを行い、svd, LDA, NMFそれぞれで50次元まで削減した特徴量）
### その他
* 作者の最初の投稿年
* storyの改行数
* コミカライズや書籍化などのキーワードフラグ
* 自動投稿のものか判断するフラグ（時間がぴったしのもの）
* keywordを分解してOneHotEncoding
* 漢字、ひらがな、カタカナ、絵文字の割合
* 投稿時刻（時間、分）
* 投稿月
* キーワードの数
* コンペ開始時からの日数
* num_code
## 集約系
* 集約特徴量（mean, std, max, min, diff)

## 効果があったもの
* 各ラベルでの２値分類予測結果  
ラベルごとに同じ特徴量で、そのラベルかどうかの２値分類に変換してlightGBMで予測したものを特徴量として追加。 
  ラベルの3,4に関しては、データ数が少ないというのもあったので、3or4とそれ以外という形の予測モデルを作成した。
  この特徴量に対して、集約処理を掛けて特徴量を増やしていった。  
  LB：0.701554 ➟ 0.675137に改善
  
* keyword でのWord2Vec  
キーワードでword2vecした特徴量を作成した。参考したのはatmacupのaraiさんのディスカッション
  https://www.guruguru.science/competitions/16/discussions/2fafef06-5a26-4d33-b535-a94cc9549ac4/
  LBとしては0.005ほどだったけど、後半での精度向上だったので比較的印象的。
  
* 自然言語処理（tfidf, BM25）  
次元圧縮とではLDAが良かったが、NMF・svdも一緒にいれたことが良かった。自然言語の処理と次元圧縮に関してはまとめて行きたい。
  
## モデル
* LightGBM（StratifiedKFold, 5fold, 3seed）  
catboostも試したが大きな変化はなかった。また前回のatmacupでも行われていた2nd stageを採用。効果はあった。
  集約特徴量も作成したが、自分のデータは除外などはしなかった。
  https://www.guruguru.science/competitions/18/discussions/dca15cb6-27a4-4e7e-836e-8613a79a3d65/
  パラメータ調整は実施セず、num_leavesとかmax_depthをいじったくらい。
  
## CV vs LB
当初からCVとLBでの乖離が問題になっていた。  
https://www.nishika.com/competitions/21/topics/156  
ここは、昔の投稿小説も学習データには存在していたので、人気になる小説はその時々の人気カテゴリに偏っていくのではないかと思ってはいた。
なので学習データを2019年、2020年以前のデータを除外してモデルを作成したところ、乖離が0.03程度までに縮めることができた。
LBの精度向上にもつながったので、良かった点でもあった。

## 上位者のSolution
今後記載予定。

## 環境構築関連でのメモ
* Stopwprdsのdownloadに関して  
'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
  をダウンロードして読み込む際に処理が上手くいかないで止まっていた。おそらくコード上で実行していたので、docker環境ではその権限関係
  で上手くいかなかったのではないかと思っている。
  

* neologdのダウンロードに関して  
Dokcerfileで以下のように設定
  ```
  USER root
  RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
    && cd mecab-ipadic-neologd \
    && bin/install-mecab-ipadic-neologd -n -y
  USER penguin
  ```
  ポイントは、一度root権限に以降してからgit cloneを行う必要があった点。ここで結構戸惑っていた。
  使っているファイルの中身を追っていく必要はあるかも。（ベースで使っているのは以下）
  https://gitlab.com/nyker510/analysis-template/-/blob/master/docker/cpu.Dockerfile  
  上はcpuの設定しかしてないので、gpu版も使ってみたい。
  
## その他メモ
* BM25Transformerに関して  
実装に関しては、ネットの広いものをそのまま使ったが、colab上で行うときにエラーが発生していた。
  結局移以下の部分の引数の問題であったが、原因は不明（versionなどの問題ではないかと思っている）
  ```python
  if self.use_idf: # ここのuse_idfを引数で持っている点が問題であった。
    n_samples, n_features = X.shape
    df = _document_frequency(X)
    idf = np.log((n_samples - df + 0.5) / (df + 0.5))
    self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
  ```

## 次元削減（圧縮）について
今回、Tfidfなどの特徴量作成に関して、次元圧縮を行っていったが、いまいち違いが分かっていないので整理中。

#### そもそも次元削減（圧縮）とは？
抽象的に言ってしまうと「元の情報を出来るだけ失わないようにコアな成分を抽出する次元削減法」のこと。
そのためには、データのばらつき（分散）が最大になるように情報を取ってくるようにする。これが主成分分析（PCA)の考え方。
分散が最大になるようにしていくってのは、データを分離しやすくなる軸（特徴）を探すって感じと思っている。

![img_1.png](img_1.png)

#### svd（特異点分解）  
自然言語で良く用いられている特異点分解。  
https://qiita.com/kidaufo/items/0f3da4ca4e19dc0e987e

