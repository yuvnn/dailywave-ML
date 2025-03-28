> **프로젝트 참고**
> 
> 
> https://github.com/lkkaram/korean-frown-sentence-classifier
> 

## 이론적 배경

### 1. Sentimental classification (감정 분류 체계)

- 폴 에크만(Paul Ekman)체계
(분노, 역겨움, 공포, 행복, 슬픔, 놀람)
- 수전 데이비드(Susan David) 체계
(분노, 슬픔, 불안, 상처, 당황, 기쁨)






---
## 사용된 호칭 명세

### 1. 라벨링

> **라벨링 변경 위치 :** ./SentimentAnalysis/constants.py
> 
- **EDC :** 0:기쁨, 1:당황, 2:분노, 3:불안, 4:상처, 5:슬픔
- **6label-1 (폴에크만) :** 0:분노, 1:역겨움, 2:공포, 3:행복, 4:슬픔, 5:놀람
- **6label-2 (수전데이비드) :** 0:분노, 1:슬픔, 2:불안, 3:상처, 4:당황, 5:기쁨
- **8label (수전데이비드+a**) : 0:분노, 1:슬픔, 2:불안, 3:상처, 4:당황, 5:기쁨, 6:감사, 7:평온




### 2. 사용된 감정 데이터 파일명

> **감정 dataset 위치 :** ./SentimentAnalysis/data/preprocess
> 
- **EDC** (EmotionalDialogCorpus)
[(감성대화말뭉치)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=86)
    - **EDC** : (EDC)
- **DVforEC** (Dialogue voice dataset for emotion classification)
[(감정분류를 위한 대화 음성 데이터셋)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263)
    - **DVforEC(5) :** Dialogue voice dataset for emotion classification (5차년도, 6label-1)
    - **DVforEC(a_6l) :** Dialogue voice dataset for emotion classification (4차년도+5차년도, 6label-1)
    - **DVforEC(4_6l) :** Dialogue voice dataset for emotion classification (4차년도, 6label-1)
    - **DVforEC(4_8l) :** Dialogue voice dataset for emotion classification (4차년도, 8label)
    - **10thou(6l) :** (라벨 당 1만개 데이터, 6lebel-2)
    - **20thou(6l) :** (라벨 당 2만개 데이터, 6lebel-2)
