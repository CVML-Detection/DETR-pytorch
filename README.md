# DETR 부수기

### official-code : https://github.com/facebookresearch/detr

### KOR code comment : https://github.com/nuggy875/note-DETR-official

### paper : https://arxiv.org/abs/2005.12872

- 하루 1시간씩 코딩

- [x] Dataset ( 조성민 )
  - [x] Nested Tensor 분석 -> resize를 600x600로 구성
  - [x] Loss에 들어가는 GT를 어떻게 변형하는지 확인 (공집합 등)
  - [ ] Data Augmentation (Random Crop)
- [x] Model ( 팽진욱 )
  - [x] 구조 이해
  - [x] Backbone (ResNet50 + Positional Encoding)
  - [x] Transformer Encoder
  - [x] Transformer Decoder
- [x] Loss (Criterion) ( 팽진욱, 조성민 )
  - [x] Matcher (scipy.optimizer, Hungarian Algorithm)
  - [x] Hungarian Loss (loss label, boxes loss, cardinality loss)
  - [ ] Check vs original loss
- [ ] Training 
  - [ ] Find training epoch, batch, lr 
