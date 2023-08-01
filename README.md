# Lg_Aimers
Lg_Aimers 온라인 해커톤 온라인 채널 제품 판매량 예측 AI 온라인 해커톤


### 코드 구조

```
${PROJECT}
├── data/
├── modules/
│   ├── datasets.py
│   └── model.py
├── results/
│   ├── serial/
├── README.md
├── train.py
└── inf.py
```

- data: 학습/추론에 필요한 데이터가 저장되는 폴더
- modules
    - datasets.py: dataset 클래스
    - models.py: 기본 lstm 모델 클래스
    - losses.py: config에서 지정한 loss function을 리턴
- train.py: 학습 시 실행하는 코드
- inf.py: 추론 시 실행하는 코드


---

### 학습 process

1. 데이터 폴더 준비
    1. 아래 구조와 같이 데이터 폴더를 생성
2. 'train CFG' 수정
3. 'python train.py' 실행
4. 'results/train/'내에 결과 (모델 가중치)가 저장됨
