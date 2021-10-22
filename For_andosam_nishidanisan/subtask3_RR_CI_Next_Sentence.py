
# %%
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#同期用コード

import pandas as pd
import numpy as np
from pyasn1.type.base import SimpleAsn1Type
from transformers import BertJapaneseTokenizer
import re

# %%
from IPython import get_ipython
import random
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, BertForNextSentencePrediction
import pytorch_lightning as pl

# %%
import codecs
from bs4 import BeautifulSoup
import unicodedata

frequent_tags= ['d', 'a', 'f', 'timex3', 't-test', 't-key', 't-val', 'm-key', 'm-val', 'r', 'cc']#抽出するタグ


'''属性考慮するタグ'''
frequent_tags_attrs= ['d_', 'd_positive', 'd_suspicious', 'd_negative', 'd_general', 'a_', 'f_', 'c_',\
                     'timex3_', 'timex3_date', 'timex3_time', 'timex3_duration', 'timex3_set', 'timex3_age', 'timex3_med', 'timex3_misc',\
                     't-test_', 't-test_executed', 't-test_negated', 't-test_other','t-key_', 't-val_',\
                     'm-key_executed', 'm-key_negated', 'm-key_other', 'm-val_', 'm-val_negated',\
                     'r_scheduled', 'r_executed', 'r_negated', 'r_other',\
                     'cc_scheduled', 'cc_executed', 'cc_negated', 'cc_other']#エンティティーのリスト
'''属性考慮しないタグ'''
frequent_tags_attrs = ['d_', 'a_', 'f_', 'timex3_', 't-test_', 't-key_', 't-val_', 'm-key_', 'm-val_', 'r_', 'cc_']

attributes_keys = ['type', 'certainty', 'state']
tags_value = [int(i) for i in range(1, len(frequent_tags_attrs)+1)]
dict_tags = dict(zip(frequent_tags_attrs, tags_value))#type_id への変換用

def entities_from_xml(file_name, attrs = True):#attrs=属性を考慮するか否か，考慮しないならFalse
    with codecs.open(file_name, "r", "utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    for elem_articles in soup.find_all("articles"):#articles内のarticleを一つずつ取り出す
        entities = []
        articles = []
        for elem in elem_articles.find_all('article'):#article内の要素を一つずつ取り出す
            entities_article = []
            text_list = []
            pos1 = 0
            pos2 = 0
            for child in elem:#取り出した要素に対して，一つずつ処理する
                #（タグのないものについても要素として取得されるので，位置(pos)はずれない）                
                text = unicodedata.normalize('NFKC', child.string)#正規化
                text = text.replace('。', '.')#句点を'.'に統一, sentenceの分割に使うため．
                pos2 += len(text)#終了位置を記憶
                if child.name in frequent_tags:#特定のタグについて，固有表現の表現形，位置，タグを取得
                    attr = ""#属性を入れるため
                    if 'type' in child.attrs:#typeがある場合には
                        attr = child.attrs['type']
                    if 'certainty' in child.attrs:#certaintyがある場合には
                        attr = child.attrs['certainty']
                    if 'state' in child.attrs:#stateがある場合には
                        attr = child.attrs['state']
                    if not attrs:#attrs=属性を考慮するか否か，考慮しないならFalse
                        attr = ""
                    entities_article.append({'name':text, 'span':[pos1, pos2],\
                        'type_id':dict_tags[str(child.name)+'_'+str(attr)],\
                        'type':str(child.name)+'_'+str(attr)})
                pos1 = pos2#次のentityの開始位置を設定
                text_list.append(text)
            if elem.name == 'article':
                article_id = elem.attrs['id']
            articles.append([article_id, "".join(text_list)])
            entities.append(entities_article) 
    return articles, entities


# %% ファイル名入力
articles, entities = entities_from_xml('MedTxt-RR-JA-training.xml', attrs=False)#属性考慮するならTrue


# %%
articles

#%%

# %%
#articleをsentenceにばらす
import re
sentences = []
for _,s in articles:
    s = [re.sub(r'\n', '', i) for i in re.split(r'\.', s)]
    s = [i for i in s if i != '']
    sentences.append(s)


# %%

# %%
id_sentence = []
for i, sentence in enumerate(sentences):
    id_sentence.append([i]*len(sentence))

# %%

# %%
def flatten(t):#リストのネストをはずす
    return [item for sublist in t for item in sublist]


# %%
def df_as_dict(df, col1, key, col2):#dfのあるカラムを辞書として使いたいとき，
    #col1のvalueがkeyになる．col2のvalueがvalueになる．
    #{df[col1]: df[col2]}みたいなイメージで使う感じ
    return df[df[col1]==str(key)][col2].values[0]

# %%
df_sentence = pd.DataFrame([flatten(id_sentence), flatten(sentences)], index=['articleID', 'text'])
df_sentence = df_sentence.T
df_sentence



# %%
df = pd.read_csv('MedTxt-RR-JA-CI-training.csv')

# %%
df_articles = pd.DataFrame(articles, columns=['articleID', 'text'])
df_articles

# %%
df_dataset = pd.concat([df_articles, df['case']], axis=1)



#  %%
case_list = []
for i in df_sentence['articleID']:
    case_list.append(df_as_dict(df_dataset, 'articleID', i, 'case'))


# %%
case_list


# %%
df_sentence['case']=case_list

# %%
df_sentence


# %%

# %%




# %%
label = df_sentence.groupby('case').size().index.values
dic_case=dict(zip(label, [i for i in range(8)]))

df_sentence['case_']=[dic_case[i] for i in df_sentence['case'].values]


# %%
df_sentence

# %%

df_sentence[df_sentence['case_']==0]
# %%
pairs_of_texts = []
for i in range(8):
    texts = df_sentence[df_sentence['case_']==i]['text'].tolist()
    for text1 in texts:
        for text2 in texts:
            pairs_of_texts.append([text1, text2, 1])

# %%
pairs_of_texts_non = []

for i in range(8):
    texts1 = df_sentence[df_sentence['case_']==i]['text'].tolist()
    texts2 = df_sentence[df_sentence['case_']!=i]['text'].tolist()
    for tx1 in texts1:
        for tx2 in texts2:
            pairs_of_texts_non.append([tx1, tx2, 0])
random.seed(42)
pairs_of_texts_non_sample = random.sample(pairs_of_texts_non, len(pairs_of_texts))



# %%
len(pairs_of_texts)
# %%
len(pairs_of_texts_non_sample )

# %%
# 日本語学習済みモデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)


# %%
"""
length_tokens = []
for i, k in df_dataset['text'].items():
    length_tokens.append(len(tokenizer.tokenize(k)))


# %%
df_dataset['len_tokens']=length_tokens


# %%
import matplotlib.pyplot as plt

df_dataset.plot(x='articleID',y='len_tokens',kind='scatter')
plt.show()
# %%
df_dataset

# %%
df_dataset['case']

# %%
# ラベル数の確認
label = df_dataset.groupby('case').size().index.values

dic_case=dict(zip(label, [i for i in range(8)]))

# %%
df_dataset['case_']=[dic_case[i] for i in df_dataset['case'].values]



# %%
df_dataset
"""


# %%
# 6-4
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
bert_sc = BertForNextSentencePrediction.from_pretrained(
    MODEL_NAME
)
bert_sc = bert_sc.cuda()




# %%
datasetList = pairs_of_texts + pairs_of_texts_non_sample
random.seed(42)
random.shuffle(datasetList) # ランダムにシャッフル
# %%
n = len(datasetList)
n_train = int(0.8*n)
n_val = int(0.1*n)
dataset_trainList = datasetList[:n_train] # 学習データ
dataset_valList = datasetList[n_train:n_train+n_val] # 検証データ
dataset_testList = datasetList[n_train+n_val:] # テストデータ






# %%
# 各データの形式を整える
def dataset_for_loader(datasetList):
    max_length = 128
    dataset_for_loader = []
    for text1, text2, label in datasetList:
        encoding = tokenizer.encode_plus(
            text1,
            text2,
            max_length=max_length, 
            padding='max_length',
            truncation=True,
        )
        encoding['labels'] = label # ラベルを追加
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset_for_loader.append(encoding)
    return dataset_for_loader


# %%
dataset_train = dataset_for_loader(dataset_trainList)
dataset_val = dataset_for_loader(dataset_valList)
dataset_test = dataset_for_loader(dataset_testList)

# %%
dataset_test

# %%
#check_length = []

#for k in range(len(dataset_test)):
#    check_length.append(len([i for i in dataset_test[k]['input_ids'] if i != 0]))

# %%
#max(check_length)

# %%
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32

g = torch.Generator()
g.manual_seed(42)


# %%
# データセットからデータローダを作成
# 学習データはshuffle=Trueにする。
dataloader_train = DataLoader(
    dataset_train, batch_size=32, worker_init_fn=seed_worker, generator=g
) 
dataloader_val = DataLoader(dataset_val, batch_size=256, worker_init_fn=seed_worker, generator=g)
dataloader_test = DataLoader(dataset_test, batch_size=256, worker_init_fn=seed_worker, generator=g)

# %%
# 6-14
class BertForNextSentencePrediction_pl(pl.LightningModule):
        
    def __init__(self, model_name, lr):
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率

        super().__init__()
        
        # 引数のnum_labelsとlrを保存。
        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters() 

        # BERTのロード
        self.bert_sc = BertForNextSentencePrediction.from_pretrained(
            model_name
        )
        
    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        return loss
        
    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss) # 損失を'val_loss'の名前でログをとる。

    # テストデータのミニバッチが与えられた時に、
    # テストデータを評価する指標を計算する関数を書く。
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # バッチからラベルを取得
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        accuracy = num_correct/labels.size(0) #精度
        self.log('accuracy', accuracy) # 精度を'accuracy'の名前でログをとる。

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# %%
# 6-15
# 学習時にモデルの重みを保存する条件を指定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model/',
)

# 学習の方法を指定
trainer = pl.Trainer(
    gpus=1, 
    max_epochs=15,
    callbacks = [checkpoint]
)


# %%
# 6-16
# PyTorch Lightningモデルのロード
model = BertForNextSentencePrediction_pl(
    MODEL_NAME, lr=1e-5
)

# %%
# ファインチューニングを行う。
trainer.fit(model, dataloader_train, dataloader_val) 


# %%
# 6-17
best_model_path = checkpoint.best_model_path # ベストモデルのファイル
print('ベストモデルのファイル: ', checkpoint.best_model_path)
print('ベストモデルの検証データに対する損失: ', checkpoint.best_model_score)


# %%
# 6-18
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir ./')

# %%
# 6-19
test = trainer.test(test_dataloaders=dataloader_test)
print(f'Accuracy: {test[0]["accuracy"]:.2f}')


# %%
model = BertForNextSentencePrediction_pl.load_from_checkpoint(best_model_path)
model = model.cuda()


# %%
model.bert_sc.save_pretrained('./model_transformers')

# %%

bert_sc = BertForNextSentencePrediction.from_pretrained(
    './model_transformers'
)
bert_sc = bert_sc.cuda()



# %%
text1_list = []
text2_list = []
label_list = []
for text1, text2, label in dataset_testList:
    text1_list.append(text1)
    text2_list.append(text2)
    label_list.append(label)

# %%
pd.DataFrame(label_list, columns=['label']).groupby('label').size()

# %%
#分類スコアを求めるために変換する関数
def encoding_plus_for_logits(dataset_List, num1, num2):#dataset_List=[text1, text2, label], num=入れすぎるとメモリ不足
    dataset_encoding_list = []
    for text1, text2, label in dataset_List:
        encoding = tokenizer.encode_plus(
            text1,
            text2,
            max_length = 128,
            padding='max_length',
            truncation=True,
        )
        encoding['labels'] = label # ラベルを追加
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset_encoding_list.append(encoding)
    #このままだとoutput = bert_sc(**dataset_encoding)に入らないので，
    #以下で整形する

    encoding_input_ids = []
    encoding_token_type = []
    encoding_attention_mask = []
    encoding_labels = []

    #dictから抜き出す
    for i in range(num1, num2):
        encoding_input_ids.append(dataset_encoding_list[i]['input_ids'])
        encoding_token_type.append((dataset_encoding_list[i]['token_type_ids']))
        encoding_attention_mask.append((dataset_encoding_list[i]['attention_mask']))
        encoding_labels.append((dataset_encoding_list[i]['labels']))

    #tensorをまとめる
    dataset_encoding = {'input_ids': torch.stack(encoding_input_ids).cuda(),
                    'token_type_ids':torch.stack(encoding_token_type).cuda(),
                    'attention_mask':torch.stack(encoding_attention_mask).cuda(),
                    'labels':torch.stack(encoding_labels).cuda()
                    }
    
    #分類ラベルを得る
    with torch.no_grad():
        output = bert_sc(**dataset_encoding)
        scores = output.logits
        labels_predicted = scores.argmax(-1)
    
    return labels_predicted


# %%
# データの符号化
labels_predicted_1 = encoding_plus_for_logits(dataset_testList, 0, 4000)
labels_predicted_2 = encoding_plus_for_logits(dataset_testList, 4000, 8000)
labels_predicted_3 = encoding_plus_for_logits(dataset_testList, 8000, len(dataset_testList))
# %%
labels_predicted_1.tolist()

# %%
labels_predicted_1

# %%
labels_predicted = labels_predicted_1.tolist() + labels_predicted_2.tolist() + labels_predicted_3.tolist()
# %%
df_test = pd.DataFrame(dataset_testList, columns=['text1', 'text2', 'label'])

# %%
df_test
# %%
#predictedを追加
df_test['predicted']=labels_predicted


# %%
def DataFrame_classification(label, predicted):
    from sklearn.metrics import classification_report
    df_eval = pd.DataFrame(classification_report(\
        label, predicted, digits=3, zero_division=0, output_dict=True)).round(3).T
    df_eval['support']=df_eval['support'].astype(int)
    return df_eval

# %%
DataFrame_classification(df_test["label"], df_test["predicted"])


# %%
#confusion_Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

train_label = df_test["label"].tolist()
train_pred = df_test["predicted"].tolist()
conf_mat = confusion_matrix(train_label, train_pred)

plot = sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt = '3g')
plot.set_xlabel("prediction")
plot.set_ylabel("label")
# %%
# %%
DataFrame_classification(df_test["label"], len(df_test)*[0])

# %%
df_test
# %%
import csv

pd.DataFrame(dataset_test).to_csv("dataset_test_RR.csv")
# %%
