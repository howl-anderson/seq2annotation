# seq2annotation

基于 TensorFlow 的通用序列标注算法库（目前包含 `BiLSTM+CRF` 和 `IDCNN+CRF`，更多算法正在持续添加中）实现中文分词（Tokenizer / segmentation）、词性标注（Part Of Speech, POS）和命名实体识别（Named Entity Recognition, NER）等序列标注任务。

## 特色
* 通用的序列标注：能够解决通用的序列标注问题：分词、词性标注和实体识别仅仅是特例。
* Tag schema free: 你可以选择你想用的任何 Tagset。依赖于 [tokenizer_tools](https://github.com/howl-anderson/tokenizer_tools) 提供的编码、解码功能
* 基于 TensorFlow Estimator: 模型代码很精干，代码量少
* 导出为 `SavedModel` 模型，可以直接使用 TensorFlow Serving 或者 `tf.predictor` API 启动

## TODO
* current [TF Metrics](https://github.com/guillaumegenthial/tf_metrics) is not launch on pypi, but seq2annotation depends on it, so seq2annotation currently can't packaged as python package on pypi

## More Algorithms To Do
* https://www.cnblogs.com/Determined22/p/7238342.html
* http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
* http://www.voidcn.com/article/p-kvrmknrl-bgh.html

## Credits
- 深受 [Guillaume Genthial](https://github.com/guillaumegenthial) 的 [tf_ner](https://github.com/guillaumegenthial/tf_ner) 项目的影响

## 增加 NER 评估方案
From http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/