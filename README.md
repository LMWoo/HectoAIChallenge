# HectoAIChallenge

## HectoAIChallenge

### 1. 대회 정보

 - [대회 페이지](https://dacon.io/competitions/official/236493/overview/description)
 - 대회 기간 : 2025.05.19 ~ 2025.06.16
 - 대회 결과 발표 : 2025.06.19
 - 주제: 중고차 이미지 차종 분류 AI 모델 개발

### 2. 대회 주제
 - 최근 자동차 산업의 디지털 전환으로 인해 정확한 차종 분류 기술의 중요성이 커지고 있으며, 이는 중고차 거래, 차량 관리, 자동 주차 등 다양한 분야에서 핵심 기술로 활용됩니다. 이에 따라 헥토(Hecto)는 2025 상반기 'HAI! - Hecto AI Challenge' 대회를 개최하여, 실제 중고차 이미지를 기반으로 한 차종 분류 AI 모델 개발을 주제로 참가자들의 창의적이고 고도화된 AI 전략을 평가하고, 우수 인재 채용과 연계하고자 합니다. - ([대회 페이지](https://dacon.io/competitions/official/236493/overview/description)의 배경 요약) 

### 3. 대회 참여 목표
 - 단순한 모델 변경 뿐만 아니라 **이론적으로만 알고 있었던 다양한 모델 성능 향상에 도움이 되는 기법**들을 실험해 보며 몸소 체득을 하는 것이 목표, 이를 통해 실전 문제 해결 능력과 깊이 있는 모델링 전략 능력을 향상시키고자 합니다.

### 3. 대회 결과
 - 1396 팀중 **168등**
   <img width="1207" alt="스크린샷 2025-06-21 오후 7 16 06" src="https://github.com/user-attachments/assets/3c340a77-62e7-4af2-8783-ba8c057bc5b2" />

### 4. 모델 실험 결과

| 실험 구성 | Pretrained model | Loss | LR Scheduler | Optimizer | 기타 기법 | Score (Log Loss) |
|-----------|--------------------|-------------|-------------------|------------|------------------------|------------------|
| 1 | ResNet50 (torchvision)   | CrossEntropy | -                 | AdamW     | -                      | **0.319**        |
| 1 | ResNet50 (timm)          | CrossEntropy | -                 | AdamW     | -                      | **0.317**        |
| 2 | ConvNeXt (timm)          | CrossEntropy | -                 | AdamW     | -                      | **0.251**        |
| 3 | ConvNeXt (timm)          | Focal Loss   | CosineAnnealing   | AdamW     | -                      | **0.195**        |
| 4 | ConvNeXt (timm)          | Focal Loss   | CosineAnnealing   | AdamW     | EMA                    | **0.185 (최종 제출)** |
| 5 | ConvNeXt (timm)          | Focal Loss   | CosineAnnealing   | AdamW     | EMA, kfold Ensemble    | **0.172 (대회 종료 후 추가 실험)*** |


### 4. 대회 후기
 - **Pretrained 모델 변경(ResNet50 → ConvNeXt)** 자체도 성능에 영향을 주는 것 뿐만 아니라 **Optimizer, Scheduler, Loss 조합**도 모델 성능 향상에 중요한 요소라는 점을 경험

 - 대회를 [Upstage MLOps](https://github.com/LMWoo/UpstageAILab_13/tree/master/MLOps) 프로젝트와 병행해서 진행하다 보니, 실험 결과를 체계적으로 기록하고 관리하지 못한 점이 아쉬움으로 남음
   - 초반에 wandb등을 사용하였으나 체계적인 기록없이 (val loss만 기록)사용하여 나중에 분석을 제대로 하지 못함, 후반에 confusion matrix 등을 도입하였고 다음 대회 때 추가 분석 기법을 도입해 다양한 분석으로 실험을 할 계획

 - TTA(Test-Time Augmentation)를 너무 단순하게 좌우 반전만 적용해, 추론 성능 향상이 거의 없었음
   - 다음 대회에서는 rotation, crop, scaling, blur 등 다양한 TTA 조합을 적용해 좀 더 강건한 예측 성능 확보를 목표로 할 예정

 - 실험 종류가 많아지면서 이전 실험에서 사용했던 파라미터들을 체계적으로 관리할 필요성을 절실히 느낌
   - 이를 해결하기 위해 hydra를 도입하여 복잡한 코드 간소화와 config기반 실험 관리 체계를 도입하여 실험할 예정

 - **실험 기록이 제일 중요**
  
### 5. 기타 대회 작업 내용
 - [EDA]([https://github.com/LMWoo/UpstageAILab_13/tree/master/MLOps](https://github.com/LMWoo/HectoAIChallenge/blob/master/notebooks/EDA.ipynb)
