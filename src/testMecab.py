# ----------------------------------------------------------------------
import sys
import MeCab
import umap
import gensim
#
mt = MeCab.Tagger("../usr/local/lib/mecab/dic/mecab-ipadic-neologd")
str_in="特急はくたかで富山に向かいます。それから、金沢に行って、兼六園に行きます。"
res = mt.parseToNode(str_in)

while res:
#    print (res.surface)
    arr = res.feature.split(",")
    if (arr[1] == "固有名詞"):
        print(res.feature)
        print(arr[6])
    res = res.next
#
# ----------------------------------------------------------------------