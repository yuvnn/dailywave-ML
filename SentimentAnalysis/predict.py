import time
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from constants import ID2LABEL_KOR
from utils import clean 

def infer(sentences):
    global model
    global tokenizer
    global device
    id2label = ID2LABEL_KOR
    results = []

    for sentence in sentences:
        sentence = clean(sentence)

        infer_stime = time.time()
        encoding = tokenizer(sentence, return_tensors='pt')
        outputs = model(**encoding)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        preds = sigmoid(logits.squeeze())
        infer_etime = time.time()

        result = {'문장': sentence,
                  '추론시간': infer_etime - infer_stime
                  }

        # 가장 높은 수치를 가진 라벨 찾기
        max_prob_idx = torch.argmax(preds).item()
        result['추론 감정'] = id2label[max_prob_idx]  # 해당 인덱스의 라벨을 가져옴

        for id, label in id2label.items():
            prob = preds[id].item()
            result[label] = prob

        results.append(result)

    results = pd.DataFrame(results)

    return results


if __name__ == '__main__':
    ckpt_path = './weights/20241114T23-38-02/checkpoint-439'
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    sentences = ['칼국수를 먹었다']

    model.eval()
    ret = infer(sentences)
    print(ret.T)
