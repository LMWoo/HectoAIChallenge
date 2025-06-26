# HectoAIChallenge

## HectoAIChallenge

### 1. 대회 정보

 - [대회 페이지](https://dacon.io/competitions/official/236493/overview/description))
 - 대회 기간 : 2025.05.19 ~ 2025.06.16
 - 대회 결과 발표 : 2025.06.19
 - 주제: 중고차 이미지 차종 분류 AI 모델 개발

### 2. 대회 목표
 - 최근 자동차 산업의 디지털 전환과 더불어, 다양한 차종을 빠르고 정확하게 인식하는 기술의 중요성이 커지고 있습니다. 특히 중고차 거래 플랫폼, 차량 관리 시스템, 자동 주차 및 보안 시스템 등 실생활에 밀접한 분야에서는 정확한 차종 분류가 핵심 기술로 떠오르고 있습니다.

이미지 기반 차종 인식 기술은 기존의 수작업 방식에 비해 높은 정확도와 효율성을 제공하며, 인공지능(AI)을 활용한 자동화 기술이 빠르게 발전하고 있습니다. 그중에서도 다양한 차종을 세밀하게 구분할 수 있는 능력은 실제 서비스 도입 시 차별화된 경쟁력을 좌우하는 요소로 작용합니다.

이에 따라, ‘HAI(하이)! - Hecto AI Challenge : 2025 상반기 헥토 채용 AI 경진대회’는 실제 중고차 차량 이미지를 기반으로 한 차종 분류 AI 모델 개발을 주제로 개최됩니다.

본 대회는 헥토의 우수 인재 발굴 및 채용 연계를 목적으로 기획된 정기적인 AI 챌린지 시리즈의 일환이며, 참가자들은 창의적이고 고도화된 AI 모델링 전략을 통해 높은 분류 성능을 달성하는 데 도전하게 됩니다.

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

 - 대회를 [Upstage MLOps](https://github.com/LMWoo/UpstageAILab_13/tree/master/MLOps)  프로젝트와 병행해서 진행하다 보니, 실험 결과를 체계적으로 기록하고 관리하지 못한 점이 아쉬움으로 남음
   -> 다음 대회에서는 wandb 등으로 실험 기록 및 분석을 할 계획

 - TTA(Test-Time Augmentation)를 너무 단순하게 좌우 반전만 적용해, 추론 성능 향상이 거의 없었음
   -> 다음 대회에서는 rotation, crop, scaling, blur 등 다양한 TTA 조합을 적용해 좀 더 강건한 예측 성능 확보를 목표로 할 예정

### 5. 대회 작업 내용
 - 
