import os
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#random seed 고정

tf.random.set_seed(1234)
np.random.seed(1234)

# BASE PARAMS

BATCH_SIZE = 32
NUM_EPOCHS = 3
MAX_LEN = 24 * 2 # Average total * 2 -> EDA에서 탐색한 결과

DATA_IN_PATH = './data_in/KOR'
DATA_OUT_PATH = "./data_out/KOR"

# 학습데이터
TRAIN_SNLI_DF = os.path.join(DATA_IN_PATH, 'KorNLI', 'snli_1.0_train.kor.tsv')
TRAIN_XNLI_DF = os.path.join(DATA_IN_PATH, 'KorNLI', 'multinli.train.ko.tsv')

train_data_snli = pd.read_csv(TRAIN_SNLI_DF, header=0, delimiter = '\t', quoting = 3)
train_data_xnli = pd.read_csv(TRAIN_XNLI_DF, header=0, delimiter = '\t', quoting = 3)

train_data_snli_xnli = train_data_snli.append(train_data_xnli) # 두개의 train data를 하나로 병함
train_data_snli_xnli = train_data_snli_xnli.dropna() # 결측치 제거
train_data_snli_xnli = train_data_snli_xnli.reset_index() # 인덱스 reset

# 검증데이터
DEV_XNLI_DF = os.path.join(DATA_IN_PATH, 'KorNLI', 'xnli.dev.ko.tsv')
dev_data_xnli = pd.read_csv(DEV_XNLI_DF, header=0, delimiter = '\t', quoting = 3)
dev_data_xnli = dev_data_xnli.dropna() # 결측치 제거

# transformers의 BertTokenizer를 이용
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased",
                                          cache_dir='bert_ckpt',
                                          do_lower_case=False)

# 2개의 문장을 받아서 전처리할 수 있는 bert_tokenizer_v2 함수를 만들어 줌
def bert_tokenizer_v2(sent1, sent2, MAX_LEN):
    encoded_dict = tokenizer.encode_plus(
        text = sent1,
        text_pair = sent2,
        add_special_tokens = True,
        max_length = MAX_LEN,
        pad_to_max_length = True,
        return_attention_mask = True)

    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']

    return input_id, attention_mask, token_type_id

# bert_tokenizer_v2의 아웃풋을 담을 수 있는 리스트를 지정
input_ids = []
attention_masks = []
token_type_ids = []

# 한 문장씩 꺼내어 bert_tokenizer_v2의 전처리 후 저장
for sent1, sent2 in zip(dev_data_xnli['sentence1'], dev_data_xnli['sentence2']):
    try:
        input_id, attention_mask, token_type_id = bert_tokenizer_v2(sent1, sent2, MAX_LEN)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)

    except Exception as e:
        pass

# label 값을 entailment, contradiction, neutral에서 0,1,2의 정수형으로 변경
label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
def convert_int(label):
    num_label = label_dict[label]
    return num_label

train_data_snli_xnli['gole_label_int'] = train_data_snli_xnli['gold_label'].spply(convert_int)
train_data_labels = np.array(train_data_snli_xnli['gole_label_int'], dtype = int)

dev_data_xnli['gold_label_int'] = dev_data_xnli['gold_label'].apply(convert_int)
dev_data_labels = np.array(dev_data_xnli['gold_label_int'], dtype = int)

print("# train labels: {}, #dev labels: {}".format(len(train_data_labels), len(dev_data_labels)))