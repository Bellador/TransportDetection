Google attention model that is able to detect French street names on street signs

# Recources
[main github repo](https://github.com/tensorflow/models/tree/master/research/attention_ocr)
[additional instructions](https://modelzoo.co/model/attentionocr)
[pretrained checkpoint](download.tensorflow.org/models/attention_ocr_2017_08_09.tar.gz)
[publication](https://arxiv.org/abs/1704.03549)
[running SavedModel for inference](https://www.tensorflow.org/tfx/serving/serving_basic)

# Remarks
folder "C:\Users\mhartman\PycharmProjects\TensorFlow\models\research\attention_ocr\python\testdata" needed to be adapted to "C:\Users\mhartman\PycharmProjects\TensorFlow\models\research\attention_ocr\python\data" due to conflict

# Commands
Export/converts existing checkpoint into a SavedModel: `python model_export.py --checkpoint=../model.ckpt-399731 --export_dir=/tmp/attention_ocr_export` 
(Hint: checkpoint was downloaded from link above and extracted)
