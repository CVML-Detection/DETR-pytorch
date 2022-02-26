# DETR 부수기

### official-code : https://github.com/facebookresearch/detr

### KOR code comment : https://github.com/nuggy875/note-DETR-official

### paper : https://arxiv.org/abs/2005.12872

- 하루 1시간씩 코딩

- [ ] Dataset ( 조성민 )
  - [ ] Nested Tensor 분석 -> resize를 600x600로 구성
  - [ ] Loss에 들어가는 GT를 어떻게 변형하는지 확인 (공집합 등)
  - [ ] Data Augmentation (Random Crop)
- [ ] Model ( 팽진욱 )
  - [x] 구조 이해
  - [ ] Backbone (ResNet50 + Positional Encoding)
  - [ ] Transformer Encoder
  - [ ] Transformer Decoder
- [ ] Loss (Criterion) ( 팽진욱, 조성민 )
  - [x] 구조 이해
  - [ ] Matcher (scipy.optimizer, Hungarian Algorithm)
  - [ ] Hungarian Loss (loss label, boxes loss, cardinality loss)
