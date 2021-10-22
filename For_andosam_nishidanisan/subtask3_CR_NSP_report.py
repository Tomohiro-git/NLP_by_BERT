
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
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
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
articles, entities = entities_from_xml('MedTxt-CR-JA-training.xml', attrs=False)#属性考慮するならTrue


# %%
articles
# %%
df = pd.read_csv('/home/is/tomohiro-ni/NTICIR16/subtask3/MedTxt-CR-JA-ADE-training.csv')

# %%
df_articles = pd.DataFrame(articles, columns=['articleID', 'text'])
df_articles

# %%
#IDを判定して，同じIDの回数分だけドキュメントを複製する．

k = 0
docList = []
for i in range(len(df)):
    ID1 = df_articles.iat[i, 0]#一方のID
    document = df_articles.iat[i,1]
    try:
        while ID1 == df.iat[k, 0]:#IDが一緒な限り，繰り返す
            docList.append(document)#ドキュメントの複製
            k += 1
    except IndexError:
        break
# %%
# 関数にしてみる
# ふたつのデータフレームに共通するデータを判定し，
# 共通する回数分だけ繰り返して複製する

def concatenate_dup(df1, col1, df2, col2, col_result):
    #df1はもとのデータフレーム
    #df2は複製したいデータがあるデータフレーム

    #col1とcol2はdf1,df2それぞれの一致している列番号
    #例えば，IDなど

    #col_resultはdf2の複製したいデータがある列番号

    k = 0
    duplicatedDataList = []
    for i in range(len(df1)):
        ID1 = df2.iat[i, col1]#一方のIDの列番号
        originalData = df2.iat[i, col_result]
        try:
            while ID1 == df1.iat[k, col2]:#IDが一緒な限り，繰り返す
                duplicatedDataList.append(originalData)#ドキュメントの複製
                k += 1
        except IndexError:
            break
    df3 = pd.DataFrame(duplicatedDataList, columns=['document'])

    return pd.concat([df1, df3], axis=1)

# %%
df_concat = concatenate_dup(df, 0, df_articles, 0, 1)

# %%
for i in range(len(df_concat)):
    text = df_concat['text'][i:i+1].to_string(index=False)
    text = unicodedata.normalize('NFKC', text)
    doc = df_concat['document'][i:i+1].to_string(index=False)
    match = re.search(re.escape(text),re.escape(doc))#escapeすると，matchが取れるのあるっぽい
    try:
        span1 = max(0, match.start())#matchが取れないの諦める
    except:
        print(doc)
        print(text)


# %%
df.groupby('ADEval').size()

# %%
# 日本語学習済みモデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)


# %%
def extract_text_from_document(text, document):
    match = re.search(re.escape(text), document)#escapeすることで，取得できないtextがなくなる
    length = len(document)

    span1 = max(0, match.start()-50)#前後50文字を取得
    span2 = min(length, match.end()+50)
    return document[span1:span2]

# %%
inputTextList = []#文章+text+tagの順で入力する
inputLabelList = []
inputEntityTxList = []#のちの評価の時ために作っておく
inputtagList = [] #のちの評価の時ために作っておく
k = 0
for i in range(len(df)):
    ID1 = df_articles.iat[i, 0]
    document = df_articles.iat[i,1]
    try:
        while ID1 == df.iat[k, 0]:
            text = unicodedata.normalize('NFKC', df.iat[k,2])#dfのtext取得（entity）
            tag = df.iat[k,1]#tag取得
            label = df.iat[k,4]
            try:
                inputTextList.append\
                    (extract_text_from_document(text, document))
                inputLabelList.append(label)
                inputEntityTxList.append(text)
                inputtagList.append(tag)
            except AttributeError:#matchで取得できないものは飛ばす
                print(document)
                print(text)
            k += 1
    except IndexError:
        break

# %%
# ラベル数の確認
pd.DataFrame(inputLabelList, columns=['label']).groupby('label').size()

# %%
# 6-4
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
bert_sc = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=4
)
bert_sc = bert_sc.cuda()


# %%
#データリストを作る
datasetList = list(map(list, (zip(*[inputTextList, inputLabelList, inputEntityTxList, inputtagList]))))



# %%
random.seed(42)
random.shuffle(datasetList) # ランダムにシャッフル
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
    for text1, label, text2, tag in datasetList:
        #encode_plusで挟んだものの"[SEP]"で 入れられるっぽい 
        encoding = tokenizer.encode_plus(
            text1,
            text2 + "[SEP]" + tag,
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



# %% シードの固定
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

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
class BertForSequenceClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率

        super().__init__()
        
        # 引数のnum_labelsとlrを保存。
        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters() 

        # BERTのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
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
    max_epochs=10,
    callbacks = [checkpoint]
)


# %%
# 6-16
# PyTorch Lightningモデルのロード
model = BertForSequenceClassification_pl(
    MODEL_NAME, num_labels=4, lr=1e-5
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
model = BertForSequenceClassification_pl.load_from_checkpoint(best_model_path)
model = model.cuda()


# %%
model.bert_sc.save_pretrained('./model_transformers')

# %%

bert_sc = BertForSequenceClassification.from_pretrained(
    './model_transformers'
)
bert_sc = bert_sc.cuda()



# %%
text_list = []
label_list = []
for text, label, EntityTx, tag in dataset_testList:
    text_list.append(text)
    label_list.append(label)

# %%
pd.DataFrame(label_list, columns=['label']).groupby('label').size()


# %%
dataset_testList

# %%
#分類スコアを求めるために変換する関数
def encoding_plus_for_logits(dataset_List, num1, num2):
    #dataset_List=[text1, text2, label], num=入れすぎるとメモリ不足, num1,num2で範囲指定する．
    dataset_encoding_list = []
    max_length = 128
    for text1, label, text2, tag in dataset_List:
        encoding = tokenizer.encode_plus(
            text1,
            text2 + "[SEP]" + tag,
            max_length=max_length, 
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
labels_predicted = encoding_plus_for_logits(dataset_testList, 0, len(dataset_testList))


# %%

labels_predicted = labels_predicted.tolist()

# %%


# %%
df_test = pd.DataFrame(dataset_testList, columns=['text', 'label', 'Entitytext', 'tag'])

# %%



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

