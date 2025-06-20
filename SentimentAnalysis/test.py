import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import DataCollatorWithPadding, logging
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from utils import multi_label_metrics, preprocess_data

import sys
sys.path.append(os.path.abspath(".."))
from constants import ID2LABEL_KOR, ID2LABEL_EN

os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
logging.set_verbosity_error()



def evaluate(opt):
    tokenizer = AutoTokenizer.from_pretrained(opt['ckpt_path'])
    # dataset = load_dataset('csv', data_files={'test': opt['test_dataset_path']})

     # 로컬 CSV 파일 로드
    test_df = pd.read_csv(opt['test_dataset_path'], dtype=str)  # 모든 열을 문자열로 로드
    # test_df.rename(columns={'document': 'sentence'}, inplace=True)


    # 데이터 타입을 확인
    print(test_df.dtypes)

    # DataFrame을 Hugging Face Dataset으로 변환
    train_dataset = Dataset.from_pandas(test_df)

    # 데이터셋 합치기
    dataset = DatasetDict({
        'test': train_dataset
    })

    dataset = dataset.map(preprocess_data,
                                  batched=True,
                                  remove_columns=dataset['test'].column_names,
                                  fn_kwargs={'tokenizer': tokenizer,
                                             'labels': list(ID2LABEL_EN.values())
                                             }
                                  )
    dataset.set_format('torch')
    dataloader = torch.utils.data.DataLoader(dataset['test'],
                                             batch_size=opt['batch_size'],
                                             shuffle=False,
                                             num_workers=opt['num_workers'],
                                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
                                             )

    scores = {'micro_f1': [],
            'roc_auc': [],
            'accuracy': []
            }
    device = torch.device(opt['device'])
    model = AutoModelForSequenceClassification.from_pretrained(opt['ckpt_path']).to(device)

    model.eval()
    for data in tqdm(dataloader, total=len(dataloader), ncols=100):
        inputs = {'input_ids': data['input_ids'].to(device),
                    'token_type_ids': data['token_type_ids'].to(device),
                    'attention_mask': data['attention_mask'].to(device)}
        labels = data['labels']
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()

        score = multi_label_metrics(logits, labels)
        scores['micro_f1'].append(score['f1'])
        scores['roc_auc'].append(score['roc_auc'])
        scores['accuracy'].append(score['accuracy'])

    micro_f1 = np.mean(scores['micro_f1'])
    roc_auc = np.mean(scores['roc_auc'])
    accuracy = np.mean(scores['accuracy'])
    print(f'micro_f1: {micro_f1:.4f}, roc_acu: {roc_auc:.4f}, accuracy: {accuracy:.4f}')



if __name__ == '__main__':
    opt = {'ckpt_path': './weights/20241113T08-39-46/checkpoint-2500',
           'test_dataset_path': './data/preprocess/DVforEC(5_6l)_test.csv',
           'device': 'cuda:0',
           'batch_size': 64,
           'num_workers': 4,
           }

    evaluate(opt)