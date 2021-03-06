
# %%
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#同期用コード

# %%
import codecs
from bs4 import BeautifulSoup
import unicodedata

frequent_tags= ['d', 'a', 'timex3', 't-test', 't-key', 't-val', 'm-key', 'm-val']#抽出するタグ
'''CRのタグ
frequent_tags_attrs= ['d_', 'd_positive', 'd_suspicious', 'd_negative', 'd_general', 'a_',\
                     'timex3_', 't-test_', 't-test_executed', 't-test_negated', 't-test_other','t-key_', 't-val_',\
                     'm-key_executed', 'm-key_negated', 'm-key_other', 'm-val_', 'm-val_negated']#エンティティーのリスト
'''


frequent_tags_attrs= ['d_', 'd_positive', 'd_suspicious', 'd_negative', 'd_general', 'a_',\
                     'timex3_', 't-test_', 't-test_executed', 't-test_negated', 't-test_other','t-key_', 't-val_',\
                     'm-key_executed', 'm-key_negated', 'm-key_other', 'm-val_', 'm-val_negated']#エンティティーのリスト
attributes_keys = ['certainty', 'state']
tags_value = [int(i) for i in range(1, len(frequent_tags_attrs)+1)]
dict_tags = dict(zip(frequent_tags_attrs, tags_value))

def entities_from_xml(file_name):
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
                text = text.replace('。', '.')#句点を'.'に統一
                pos2 += len(text)#終了位置を記憶
                if child.name in frequent_tags:#特定のタグについて，固有表現の表現形，位置，タグを取得
                    attr = ""#属性を入れるため
                    if 'certainty' in child.attrs:#certaintyがある場合には
                        attr = child.attrs['certainty']
                    if 'state' in child.attrs:#stateがある場合には
                        attr = child.attrs['state']  
                    entities_article.append({'name':text, 'span':[pos1, pos2], 'type_id':dict_tags[str(child.name)+'_'+str(attr)], 'type':str(child.name)+'_'+str(attr)})
                pos1 = pos2#次のentityの開始位置を設定
                text_list.append(text)
            articles.append("".join(text_list))
            entities.append(entities_article) 
    return articles, entities

# %% ファイル名入力
articles, entities = entities_from_xml('MedTxt-RR-JA-training.xml')
# %%
entities

# %%
#articleをsentenceにばらす
import re
sentences = []
for s in articles:
    sentences.append(re.split(r'.\n', s))#多分もっといい分け方がある...一部きちんとわかれない箇所あり

# %%
# 文単位にばらしたものに，エンティティを付与し直す
texts_dataset = []
for i in range(len(sentences)):
    pos = 0
    text_dataset = []
    for k in range(len(sentences[i])):
        text_dataset.append({'ID': [i, k]})#IDを追加してみる，元に戻す時用, [何番目のarticleか，何番目のsentenceか]
        text_dataset[k].update({'text': sentences[i][k]})#テキスト追加
        tmp_entities = []
        while entities[i][0]['span'][1] <= len(sentences[i][k]) + pos:#終了位置が超えていたら，次の文へ
            entities[i][0]['span'] = [entities[i][0]['span'][0] - pos,\
                entities[i][0]['span'][1] - pos]#span入力
            tmp_entities.append(entities[i][0])
            del entities[i][0]#entity入れたら消していく
            if not entities[i]:#entitiyがなくなったら終わり
                break
        text_dataset[k].update({'entities': tmp_entities})        
        if not entities[i]:#entitiyがなくなったら終わり
            break
        pos += len(sentences[i][k])+2#'\n.'でスプリットしているので+2
    texts_dataset.append(text_dataset)

# %%
# ネストをはずす
dataset_t = []
for i in texts_dataset:
    dataset_t.extend(i)
dataset_t

# %%
# textが空のもの削除 
# 暫定でやっているが，おそらく文単位のばらしがうまくいっていない箇所あり．
# （本来はsplitのときに使う正規表現を精査するべきだが，影響が小さそうなのでとりあえずこのまま
dataset = []
for d in dataset_t:
    if d['text']!="":
        dataset.append(d)
dataset
# %%
# 8-3
import itertools
import random
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import unicodedata

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForTokenClassification
import pytorch_lightning as pl




# %%
# 8-16
# PyTorch Lightningのモデル
class BertForTokenClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_tc = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    def training_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



# %%
# 8-21
class NER_tokenizer_BIO(BertJapaneseTokenizer):

    # 初期化時に固有表現のカテゴリーの数`num_entity_type`を
    # 受け入れるようにする。
    def __init__(self, *args, **kwargs):
        self.num_entity_type = kwargs.pop('num_entity_type')
        super().__init__(*args, **kwargs)

    def encode_plus_tagged(self, text, entities, max_length):
        """
        文章とそれに含まれる固有表現が与えられた時に、
        符号化とラベル列の作成を行う。
        """
        # 固有表現の前後でtextを分割し、それぞれのラベルをつけておく。
        splitted = [] # 分割後の文字列を追加していく
        position = 0
        for entity in entities:
            start = entity['span'][0]
            end = entity['span'][1]
            label = entity['type_id']
            splitted.append({'text':text[position:start], 'label':0})
            splitted.append({'text':text[start:end], 'label':label})
            position = end
        splitted.append({'text': text[position:], 'label':0})
        splitted = [ s for s in splitted if s['text'] ]

        # 分割されたそれぞれの文章をトークン化し、ラベルをつける。
        tokens = [] # トークンを追加していく
        labels = [] # ラベルを追加していく
        for s in splitted:
            tokens_splitted = self.tokenize(s['text'])
            label = s['label']
            if label > 0: # 固有表現
                # まずトークン全てにI-タグを付与
                labels_splitted =  \
                    [ label + self.num_entity_type ] * len(tokens_splitted)
                # 先頭のトークンをB-タグにする
                labels_splitted[0] = label
            else: # それ以外
                labels_splitted =  [0] * len(tokens_splitted)
            
            tokens.extend(tokens_splitted)
            labels.extend(labels_splitted)

        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.convert_tokens_to_ids(tokens)
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length',
            truncation=True
        ) 

        # ラベルに特殊トークンを追加
        labels = [0] + labels[:max_length-2] + [0]
        labels = labels + [0]*( max_length - len(labels) )
        encoding['labels'] = labels

        return encoding

    def encode_plus_untagged(
        self, text, max_length=None, return_tensors=None
    ):
        """
        文章をトークン化し、それぞれのトークンの文章中の位置も特定しておく。
        IO法のトークナイザのencode_plus_untaggedと同じ
        """
        # 文章のトークン化を行い、
        # それぞれのトークンと文章中の文字列を対応づける。
        tokens = [] # トークンを追加していく。
        tokens_original = [] # トークンに対応する文章中の文字列を追加していく。
        words = self.word_tokenizer.tokenize(text) # MeCabで単語に分割
        for word in words:
            # 単語をサブワードに分割
            tokens_word = self.subword_tokenizer.tokenize(word) 
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]': # 未知語への対応
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##','') for token in tokens_word
                ])

        # 各トークンの文章中での位置を調べる。（空白の位置を考慮する）
        position = 0
        spans = [] # トークンの位置を追加していく。
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break

        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.convert_tokens_to_ids(tokens) 
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length' if max_length else False, 
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        # 特殊トークン[CLS]に対するダミーのspanを追加。
        spans = [[-1, -1]] + spans[:sequence_length-2] 
        # 特殊トークン[SEP]、[PAD]に対するダミーのspanを追加。
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) ) 

        # 必要に応じてtorch.Tensorにする。
        if return_tensors == 'pt':
            encoding = { k: torch.tensor([v]) for k, v in encoding.items() }

        return encoding, spans

    @staticmethod
    def Viterbi(scores_bert, num_entity_type, penalty=10000):
        """
        Viterbiアルゴリズムで最適解を求める。
        """
        m = 2*num_entity_type + 1
        penalty_matrix = np.zeros([m, m])
        for i in range(m):
            for j in range(1+num_entity_type, m):
                if not ( (i == j) or (i+num_entity_type == j) ): 
                    penalty_matrix[i,j] = penalty
        path = [ [i] for i in range(m) ]
        scores_path = scores_bert[0] - penalty_matrix[0,:]
        scores_bert = scores_bert[1:]

        

        for scores in scores_bert:
            assert len(scores) == 2*num_entity_type + 1
            score_matrix = np.array(scores_path).reshape(-1,1) \
                + np.array(scores).reshape(1,-1) \
                - penalty_matrix
            scores_path = score_matrix.max(axis=0)
            argmax = score_matrix.argmax(axis=0)
            path_new = []
            for i, idx in enumerate(argmax):
                path_new.append( path[idx] + [i] )
            path = path_new

        labels_optimal = path[np.argmax(scores_path)]
        return labels_optimal

    def convert_bert_output_to_entities(self, text, scores, spans):
        """
        文章、分類スコア、各トークンの位置から固有表現を得る。
        分類スコアはサイズが（系列長、ラベル数）の2次元配列
        """
        assert len(spans) == len(scores)
        num_entity_type = self.num_entity_type
        
        # 特殊トークンに対応する部分を取り除く
        scores = [score for score, span in zip(scores, spans) if span[0]!=-1]
        spans = [span for span in spans if span[0]!=-1]

        # Viterbiアルゴリズムでラベルの予測値を決める。
        labels = self.Viterbi(scores, num_entity_type)

        # 同じラベルが連続するトークンをまとめて、固有表現を抽出する。
        entities = []
        for label, group \
            in itertools.groupby(enumerate(labels), key=lambda x: x[1]):
            
            group = list(group)
            start = spans[group[0][0]][0]
            end = spans[group[-1][0]][1]

            if label != 0: # 固有表現であれば
                if 1 <= label <= num_entity_type:
                     # ラベルが`B-`ならば、新しいentityを追加
                    entity = {
                        "name": text[start:end],
                        "span": [start, end],
                        "type_id": label
                    }
                    entities.append(entity)
                else:
                    # ラベルが`I-`ならば、直近のentityを更新
                    entity['span'][1] = end 
                    entity['name'] = text[entity['span'][0]:entity['span'][1]]
                
        return entities



# %%
# 日本語学習済みモデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = NER_tokenizer_BIO.from_pretrained(MODEL_NAME, num_entity_type=len(frequent_tags_attrs))#Entityの数を変え忘れないように！！

# %%
# データセットの分割(8:2に分割)
random.seed(42)
random.shuffle(dataset)
n = len(dataset)
n_train = int(n*0.8)
dataset_train = dataset[:n_train]
dataset_test = dataset[n_train:]

# %%
#クロスバリデーション

from sklearn.model_selection import KFold

def create_dataset_for_CV(dataset, n_splits):
    kf = KFold(n_splits = n_splits,random_state=42, shuffle = True)
    #kFoldで取得できるのはインデックスのみ

    train_kFold_list = []
    val_kFold_list = []
    df_dataset = pd.DataFrame(dataset)# DataFrame挟んだ方がilocで一括指定できるので，dfへ変換する

    for train_index, val_index in kf.split(df_dataset):
        df_train = df_dataset.iloc[train_index]#ilocでtrainを一括指定
        df_val = df_dataset.iloc[val_index]

        # データフレームの展開（create_dataset()に合わせる形に変形）
        train_to_dict = df_train.to_dict(orient='list')
        val_to_dict = df_val.to_dict(orient='list')

        dataset_train = []
        for text, entities in zip(train_to_dict['text'], train_to_dict['entities']):#text, entitiesをzipして，対応づける形に戻す
            dataset_train.append({'text': text, 'entities': entities})
        
        dataset_val = []
        for text, entities in zip(val_to_dict['text'], val_to_dict['entities']):#text, entitiesをzipして，対応づける形に戻す
            dataset_val.append({'text': text, 'entities': entities})
        
        train_kFold_list.append(dataset_train)#分割分をリストに追加
        val_kFold_list.append(dataset_val)
    return train_kFold_list, val_kFold_list
train_kFold_list, val_kFold_list = create_dataset_for_CV(dataset_train, 5)


# %%
def create_dataset(tokenizer, dataset, max_length):
    """
    データセットをデータローダに入力できる形に整形。
    """
    dataset_for_loader = []
    for sample in dataset:
        text = sample['text']
        entities = sample['entities']
        encoding = tokenizer.encode_plus_tagged(
            text, entities, max_length=max_length
        )
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset_for_loader.append(encoding)
    return dataset_for_loader

# %%

#クロスバリデーション（k=5）
for i in range(5):
    # 8-22
    # トークナイザのロード
    # 日本語学習済みモデル
    MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    # 固有表現のカテゴリーの数`num_entity_type`を入力に入れる必要がある。
    tokenizer = NER_tokenizer_BIO.from_pretrained(
        MODEL_NAME,
        num_entity_type=len(frequent_tags_attrs) #Entityの数を変え忘れないように！
    )

    dataset_train = train_kFold_list[i]
    dataset_val = val_kFold_list[i]
    # データセットの作成
    max_length = 128
    dataset_train_for_loader = create_dataset(
        tokenizer, dataset_train, max_length
    )
    dataset_val_for_loader = create_dataset(
        tokenizer, dataset_val, max_length
    )

    # データローダの作成
    dataloader_train = DataLoader(
        dataset_train_for_loader, batch_size=32, shuffle=True
    )
    dataloader_val = DataLoader(dataset_val_for_loader, batch_size=256)

    # 8-23

    # ファインチューニング
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath='model_BIO/'
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5,
        callbacks=[checkpoint]
    )

    # PyTorch Lightningのモデルのロード
    num_entity_type = len(frequent_tags_attrs)#entityの数を変え忘れないように！！
    num_labels = 2*num_entity_type+1
    model = BertForTokenClassification_pl(
        MODEL_NAME, num_labels=num_labels, lr=1e-5
    )

    trainer.fit(model, dataloader_train, dataloader_val)
    trainer.save_checkpoint(str(i)+".ckpt")



# %%
# 性能評価

result = []
#クロスバリデーション（k=5）
for i in range(5):
    model = BertForTokenClassification_pl.load_from_checkpoint(
        checkpoint_path=str(i)+".ckpt"
    ) 
    bert_tc = model.bert_tc.cuda()

    entities_list = [] # 正解の固有表現を追加していく
    entities_predicted_list = [] # 抽出された固有表現を追加していく
    text_entities = []
    for sample in tqdm(dataset_test):
        text = sample['text']
        encoding, spans = tokenizer.encode_plus_untagged(
            text, return_tensors='pt'
        )
        encoding = { k: v.cuda() for k, v in encoding.items() } 
        
        with torch.no_grad():
            output = bert_tc(**encoding)
            scores = output.logits
            scores = scores[0].cpu().numpy().tolist()
            
        # 分類スコアを固有表現に変換する
        entities_predicted = tokenizer.convert_bert_output_to_entities(
            text, scores, spans
        )

        entities_list.append(sample['entities'])
        entities_predicted_list.append(entities_predicted)
        text_entities.append({'text': text, 'entities': sample['entities'], 'entities_predicted': entities_predicted})


    # 8-19
    def evaluate_model(entities_list, entities_predicted_list, type_id=None):
        """
        正解と予測を比較し、モデルの固有表現抽出の性能を評価する。
        type_idがNoneのときは、全ての固有表現のタイプに対して評価する。
        type_idが整数を指定すると、その固有表現のタイプのIDに対して評価を行う。
        """
        num_entities = 0 # 固有表現(正解)の個数
        num_predictions = 0 # BERTにより予測された固有表現の個数
        num_correct = 0 # BERTにより予測のうち正解であった固有表現の数

        # それぞれの文章で予測と正解を比較。
        # 予測は文章中の位置とタイプIDが一致すれば正解とみなす。
        for entities, entities_predicted \
            in zip(entities_list, entities_predicted_list):

            if type_id:
                entities = [ e for e in entities if e['type_id'] == type_id ]
                entities_predicted = [ 
                    e for e in entities_predicted if e['type_id'] == type_id
                ]
                
            get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
            set_entities = set( get_span_type(e) for e in entities )
            set_entities_predicted = \
                set( get_span_type(e) for e in entities_predicted )

            num_entities += len(entities)
            num_predictions += len(entities_predicted)
            num_correct += len( set_entities & set_entities_predicted )

        # 指標を計算
        precision = num_correct/num_predictions # 適合率
        recall = num_correct/num_entities # 再現率
        f_value = 2*precision*recall/(precision+recall) # F値

        result = {
            'num_entities': num_entities,
            'num_predictions': num_predictions,
            'num_correct': num_correct,
            'precision': precision,
            'recall': recall,
            'f_value': f_value
        }

        return result

    print(evaluate_model(entities_list, entities_predicted_list))
    result.append(evaluate_model(entities_list, entities_predicted_list))
# %%
precision = [r['precision'] for r in result]
recall = [r['recall'] for r in result]
f_value = [r['f_value'] for r in result]

# %%
print('f_value', f_value)
print('precision:{}'.format(np.average(precision)))
print('recall:{}'.format(np.average(recall)))
print('f_value:{}'.format(np.average(f_value)))


# %%
#5で評価する
model = BertForTokenClassification_pl.load_from_checkpoint(
    checkpoint_path=str(4)+".ckpt"
) 
bert_tc = model.bert_tc.cuda()

entities_list = [] # 正解の固有表現を追加していく
entities_predicted_list = [] # 抽出された固有表現を追加していく
text_entities = []
for sample in tqdm(dataset_test):
    text = sample['text']
    encoding, spans = tokenizer.encode_plus_untagged(
        text, return_tensors='pt'
    )
    encoding = { k: v.cuda() for k, v in encoding.items() } 
    
    with torch.no_grad():
        output = bert_tc(**encoding)
        scores = output.logits
        scores = scores[0].cpu().numpy().tolist()
        
    # 分類スコアを固有表現に変換する
    entities_predicted = tokenizer.convert_bert_output_to_entities(
        text, scores, spans
    )

    entities_list.append(sample['entities'])
    entities_predicted_list.append(entities_predicted)
    text_entities.append({'text': text, 'entities': sample['entities'], 'entities_predicted': entities_predicted})

# %%
# 8-19
def evaluate_model(entities_list, entities_predicted_list, type_id=None):
    """
    正解と予測を比較し、モデルの固有表現抽出の性能を評価する。
    type_idがNoneのときは、全ての固有表現のタイプに対して評価する。
    type_idが整数を指定すると、その固有表現のタイプのIDに対して評価を行う。
    """
    num_entities = 0 # 固有表現(正解)の個数
    num_predictions = 0 # BERTにより予測された固有表現の個数
    num_correct = 0 # BERTにより予測のうち正解であった固有表現の数

    # それぞれの文章で予測と正解を比較。
    # 予測は文章中の位置とタイプIDが一致すれば正解とみなす。
    for entities, entities_predicted \
        in zip(entities_list, entities_predicted_list):

        if type_id:
            entities = [ e for e in entities if e['type_id'] == type_id ]
            entities_predicted = [ 
                e for e in entities_predicted if e['type_id'] == type_id
            ]
            
        get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
        set_entities = set( get_span_type(e) for e in entities )
        set_entities_predicted = \
            set( get_span_type(e) for e in entities_predicted )

        num_entities += len(entities)
        num_predictions += len(entities_predicted)
        num_correct += len( set_entities & set_entities_predicted )

    precision = 0
    recall = 0
    f_value = 0
    # 指標を計算
    if num_predictions != 0:
        precision = num_correct/num_predictions # 適合率
    if num_entities != 0:
        recall = num_correct/num_entities # 再現率
    if precision+recall != 0:
        f_value = 2*precision*recall/(precision+recall) # F値

    try:
        result = {
            'type_id': type_id,
           'num_entities': num_entities,
            'num_predictions': num_predictions,
            'num_correct': num_correct,
            'precision': precision,
            'recall': recall,
            'f_value': f_value
        }
    except:
        pass

    return result

# %%
evaluate =[]
for i in range(1, len(frequent_tags_attrs)+1):
    evaluate.append(evaluate_model(entities_list, entities_predicted_list, type_id=i))
# %%
df_eval = pd.DataFrame(evaluate)
df_eval
# %%
df_eval.insert(0, 'type', dict_tags)
# %%
df_eval

# %%
[i['entities'] for i in text_entities]
# %%
