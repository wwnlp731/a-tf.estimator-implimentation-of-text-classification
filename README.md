# A-tf.estimator-implimentation-of-text-classification
A Tensorflow pipeline for text classification task. the model is based on Text CNN with attention and position encoder. A pipeline is created: raw_data --> tf.record --> tf.dataset -- distributed tf.esitmator.
## trainning and evalutate process:
1. Make a instance of Dataset and generate tf_record files from raw data. This step is not in pipeline. You can import Dataset.class in another python file to do this.
2. Once you have tf.record files ready, just run distributed_esitmator.py. default config is used for training and exporting.
## exporting your saved_model
1. once you finish training, esitmator you auto export best model, just load this model so that you can predict.
