from tensorflow.keras.layers import *
import tensorflow as tf
import pickle
import numpy as np
from collections import Counter
from tqdm import *
import sys
import os


class Config(object):
    embedding_dim = 100
    num_classes = 26
    num_filters = 256
    filter_sizes_w = [3, 2, 1]
    filter_sizes_c = [4, 3, 2]
    vocab_size = 100000  # 词汇表达小
    hidden_dim = 256  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    input_keep_prob = 0.5  ## 随机mask
    learning_rate = 5e-4  # 学习率
    batch_size = 1024  # 每批训练大小
    num_epochs = 1000  # 总迭代轮次
    smoothing_rate = 0.05

    use_pretrain_embs = False
    use_pe = True
    is_tfrecord = False

    penal_parms = 1e-4
    kn_num = 8
    clip_by_norm = 10
    attention_hoop = 10
    max_padding = [64, 100]
    mode = 'train'
    delimit_token = '||'

    base_dir = 'base_dir'
    train_path = 'train_path'
    eval_path = 'eval_path'
    test_path = 'test_path'
    tf_record_train = 'tf_record_train'
    tf_record_eval = 'tf_record_eval'
    tf_record_test = 'tf_record_test'
    embedding_path = 'Embedding_path'
    vocab_path = 'vocab_path'
    label_path = 'label_path'
    fix_label_path = 'fix_label_path'


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.mode = self.config.mode

        if self.mode == 'train':
            self.file_path = self.config.train_path
            self.tfrecordfile = self.config.tf_record_train
        elif self.mode == 'eval':
            self.file_path = self.config.eval_path
            self.tfrecordfile = self.config.tf_record_eval
        elif self.mode == 'test':
            self.file_path = self.config.test_path
            self.tfrecordfile = self.config.tf_record_test
        else:
            raise Exception('请输入 train, eval, test中的一个')
        self.Embedding_path = self.config.Embedding_path
        self.num_epochs = self.config.num_epochs
        self.batch_size = self.config.batch_size
        # self.is_shuffle = self.config.is_shuffle
        self.use_pretrain_embs = self.config.use_pretrain_embs
        self.vocab_size = self.config.vocab_size
        # print('mode:', self.mode)

        self.__build()

    def __build(self):
        if self.config.use_pretrain_embs == True:
            self.word2id, _ = self._get_id_embs()
        elif self.use_pretrain_embs == False and self.mode == 'train':
            self._build_vocab()
        self.word2id, self.id2word, self.label2id, self.id2label = self._read_vocab()
        if self.config.is_tfrecord and not os.path.join(self.tfrecordfile):
            self._generate_tfrecords()

    def _read_file(self, origin):
        chars, contents, labels = [], [], []
        with open(origin, 'r')as f:
            lines = f.readlines()
        for line in lines:
            q = line.split('\t')
            if len(q) != 3:
                continue
            content, label = q[0], q[1].strip()
            content = content.lower()
            content = content.strip().split()
            char = [list(i) for i in content if i != '<number>']
            char = [i for j in char for i in j]
            chars += [char]
            # contents += [content + [delimit_token] + char]
            contents += [content]
            labels += [label]
        return contents, chars, labels

    def _build_vocab(self):
        contents, chars, labels = self._read_file(self.file_path)
        content_list = [i for j in contents for i in j]
        char_list = [i for j in chars for i in j]
        comb_list = content_list + char_list
        word_count = Counter(comb_list)
        most_common_word = word_count.most_common(self.vocab_size)
        most_common_word, _ = list(zip(*most_common_word))
        labels = Counter(labels).most_common(self.config.num_classes)
        labels, _ = list(zip(*labels))
        # labels = list(labels)
        # labels.sort() ##保持label顺序
        # print(labels)
        check = set(most_common_word)
        if '<number>' not in check:
            word_list = ['<pad>'] + ['<unk>'] + ['<number>'] + list(most_common_word)
        else:
            word_list = ['<pad>'] + ['<unk>'] + list(most_common_word)
        with open(self.config.vocab_path, 'w')as fw:
            for idx, v in enumerate(word_list):
                fw.write(v + '\t' + str(idx) + '\n')
        with open(self.config.label_path, 'w')as fw:
            for idx, v in enumerate(labels):
                fw.write(v + '\t' + str(idx) + '\n')

    def _read_vocab(self):
        with open(self.config.vocab_path, 'r')as f:
            words = f.readlines()
        words = [i.split('\t')[0] for i in words]
        word2id = dict([(v, idx) for idx, v in enumerate(words)])
        id2word = dict([(idx, v) for idx, v in enumerate(words)])

        with open(self.config.fix_label_path, 'r')as f:
            labels = f.readlines()
        labels = [i.split('\t')[0] for i in labels]
        label2id = dict([(v, idx) for idx, v in enumerate(labels)])
        print(label2id)
        id2label = dict([(idx, v) for idx, v in enumerate(labels)])
        return word2id, id2word, label2id, id2label

    def _sent2id(self):
        contents, chars, labels = self._read_file(self.file_path)
        '''返回三个迭代器'''
        word2id_iter = map(lambda x: [self.word2id.get(i, 1) for i in x], contents)
        char2id_iter = map(lambda x: [self.word2id.get(i, 1) for i in x], chars)
        label2id_iter = map(lambda i: self.label2id.get(i, 'unk_label'), labels)
        return word2id_iter, char2id_iter, label2id_iter

    def _generate_tfrecords(self):
        sequences, chars, labels = [], [], []
        word2id_iter, char2id_iter, label2id_iter = self._sent2id()
        while True:
            try:
                sequence = next(word2id_iter)
                if len(sequence) > self.config.max_padding[0]:
                    sequence = sequence[:self.config.max_padding[0]]
                sequences += [sequence]
            except:
                break
        while True:
            try:
                char = next(char2id_iter)
                if len(char) > self.config.max_padding[1]:
                    char = char[:self.config.max_padding[1]]
                chars += [char]
            except:
                break
        while True:
            try:
                labels += [next(label2id_iter)]
            except:
                break
        # return input_x, input_y
        print('length_x:{}, length_y:{}'.format(len(sequences), len(labels)))
        with tf.python_io.TFRecordWriter(self.tfrecordfile) as f:
            for feature, char, label in tqdm(zip(sequences, chars, labels)):
                word_feature = list(
                    map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), feature))
                char_feature = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), char))
                example = tf.train.SequenceExample(
                    context=tf.train.Features(feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}),
                    feature_lists=tf.train.FeatureLists(feature_list={
                        'sequence': tf.train.FeatureList(feature=word_feature),
                        'chars': tf.train.FeatureList(feature=char_feature)
                    })
                )
                f.write(example.SerializeToString())

    def _single_example_parser(self, serialized_example):
        context_features = {
            "label": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "sequence": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "chars": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        labels = context_parsed['label']
        sequences = sequence_parsed
        return sequences, labels

    def _batched_data(self, path, buffer_size=1000000):
        if path == self.config.tf_record_eval or path == self.config.tf_record_test:
            self.num_epochs = 1
        print('self.num_epochs', self.num_epochs)
        dataset = tf.data.TFRecordDataset(path, num_parallel_reads=32) \
            .map(self._single_example_parser) \
            .padded_batch(int(self.batch_size / 4), padded_shapes=({'sequence': [None], 'chars': [None]}, [])) \
            .shuffle(buffer_size) \
            .repeat(self.num_epochs) \
            .prefetch(buffer_size)
        return dataset
        '''
        if path == self.config.tf_record_train:
            return dataset
        else:
            print('use_dataset.iterator!!!')
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            features = {'sentence':features}
            return features, labels
        '''

    def _get_id_embs(self):
        with open(self.Embedding_path, 'rb') as f:
            a = pickle.load(f)
            word2id = a[0]
            embs = a[1]
        return word2id, embs


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def _attention(self, input_x, hidden_dim, hoop=4):
        sw = input_x
        # print('sw',sw)
        sw_ = tf.nn.tanh(tf.layers.dense(sw, hidden_dim))
        sw_ = tf.layers.dense(sw_, hoop)
        # print('sw_', sw_)
        weight = tf.transpose(tf.nn.softmax(sw_, -1), perm=[0, 2, 1])
        # print('weight', weight)
        weighted_sw = tf.matmul(weight, sw)
        weighted_sw = tf.reshape(weighted_sw, [-1, hoop * int(sw.get_shape()[-1])])
        return weight, weighted_sw

    def _position_encoder(self, input_x, PE_dims=50, period=10000, scale=False):
        pos_ta = tf.TensorArray(size=0, dtype=tf.int32, dynamic_size=True)
        init_state = (0, pos_ta)
        condition = lambda i, _: i < tf.shape(input_x)[0]
        body = lambda i, pos_ta: (i + 1, pos_ta.write(i, tf.range(tf.shape(input_x)[1])))
        _, pos_ta = tf.while_loop(condition, body, init_state)
        pos_ta_final_result = pos_ta.stack()

        # First part of the PE function: sin and cos argument
        # First part of the PE function: sin and cos argument

        position_enc = np.array([
            [pos / np.power(period, 2 * i / PE_dims) for i in range(PE_dims)]
            for pos in range(300)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)
        lookup_table = tf.cast(lookup_table, tf.float32)
        outputs = tf.nn.embedding_lookup(lookup_table, pos_ta_final_result)
        if scale:
            outputs = outputs * PE_dims ** 0.5
            print('Position encoder scaled')
        # print(outputs)
        return outputs

    def _label_smoothing(self, one_hot_labels, epsilon=0.1):
        K = one_hot_labels.get_shape().as_list()[-1]  ## get last dimension
        return ((1 - epsilon) * one_hot_labels) + (epsilon / K)

    def _tile_eye(self, A):
        A_T = tf.transpose(A, perm=[0, 2, 1])
        AA_T = tf.matmul(A, A_T)
        eye = tf.eye(int(AA_T.get_shape()[1]))
        tile_eye_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
        init_state = (0, tile_eye_ta)
        condition = lambda i, _: i < tf.shape(A)[0]
        body = lambda i, tile_eye_ta: (i + 1, tile_eye_ta.write(i, eye))
        _, tile_eye_ta = tf.while_loop(condition, body, init_state)
        tile_eye = tile_eye_ta.stack()
        return tile_eye, AA_T

    def call(self, config, input_x, input_y, mode, input_keep_prob, dropout_keep_prob):
        '''input_x: dict, input_y = array'''
        print('input_keep:{},dropout_keep:{}'.format(input_keep_prob, dropout_keep_prob))
        conv_w1 = Conv1D(config.num_filters, config.filter_sizes_w[0],
                         padding='same', activation=None)
        conv_w2 = Conv1D(config.num_filters, config.filter_sizes_w[1],
                         padding='same', activation=None)
        conv_w3 = Conv1D(config.num_filters, config.filter_sizes_w[2],
                         padding='same', activation=None)

        conv_c1 = Conv1D(config.num_filters, config.filter_sizes_c[0],
                         padding='same', activation=None)
        conv_c2 = Conv1D(config.num_filters, config.filter_sizes_c[1],
                         padding='same', activation=None)
        conv_c3 = Conv1D(config.num_filters, config.filter_sizes_c[2],
                         padding='same', activation=None)
        # input_x = tf.constant(input_x,dtype=tf.int64)
        y_copy = input_y

        '''parse input_x'''
        x_chars = input_x['chars']
        x_words = input_x['sequence']

        if config.use_pretrain_embs == False:
            embedding = Embedding(input_dim=config.vocab_size, output_dim=config.embedding_dim)
            x_w = embedding(x_words)
            x_w = tf.nn.dropout(x_w, input_keep_prob)
            x_c = embedding(x_chars)
            x_c = tf.nn.dropout(x_c, input_keep_prob)
        else:
            raise Exception('预训练embedding_lookup还没写')
        if config.use_pe == True:
            with tf.variable_scope('Position_encoder_w', reuse=False):
                pos_encoder_w = self._position_encoder(input_x=x_words, PE_dims=50, period=1000, scale=False)
                x_w = tf.layers.dense(tf.concat([x_w, pos_encoder_w], -1), config.embedding_dim)
            with tf.variable_scope('Position_encoder_c', reuse=False):
                pos_encoder_c = self._position_encoder(input_x=x_chars, PE_dims=50, period=1000, scale=False)
                x_c = tf.layers.dense(tf.concat([x_c, pos_encoder_c], -1), config.embedding_dim)
            print('use Position encoder == True')
        with tf.variable_scope('Convolutional_block'):
            channel_w1 = tf.nn.dropout(conv_w1(x_w), dropout_keep_prob)
            channel_w2 = tf.nn.dropout(conv_w2(x_w), dropout_keep_prob)
            channel_w3 = tf.nn.dropout(conv_w3(x_w), dropout_keep_prob)

            channel_c1 = tf.nn.dropout(conv_c1(x_c), dropout_keep_prob)
            channel_c2 = tf.nn.dropout(conv_c2(x_c), dropout_keep_prob)
            channel_c3 = tf.nn.dropout(conv_c3(x_c), dropout_keep_prob)
        pooled_w = tf.concat([channel_w1, channel_w2, channel_w3], -1)  # batch_size * seq_length_w * number_filters
        pooled_c = tf.concat([channel_c1, channel_c2, channel_c3], -1)  # batch_size * seq_length_c * number_filters

        pooled_w = tf.nn.dropout(pooled_w, dropout_keep_prob)
        pooled_c = tf.nn.dropout(pooled_c, dropout_keep_prob)
        with tf.variable_scope('self_attention_w'):
            A_w, pooled_w = self._attention(pooled_w, config.hidden_dim, hoop=config.attention_hoop)
        with tf.variable_scope('self_attention_c'):
            A_c, pooled_c = self._attention(pooled_c, config.hidden_dim, hoop=config.attention_hoop)
        pooled = tf.concat([pooled_w, pooled_c], -1)
        with tf.variable_scope('predict'):
            fc1 = tf.layers.dense(pooled, config.hidden_dim, name='fc1', activation='relu')
            fc1 = tf.nn.dropout(fc1, dropout_keep_prob)
            logits = tf.layers.dense(fc1, config.num_classes, name='logits')
            softmax_score = tf.nn.softmax(logits, name='softmax_score')
            pred = tf.argmax(logits, -1, name='predict')

        if mode == tf.estimator.ModeKeys.PREDICT:
            return softmax_score
        with tf.variable_scope('metrics'):
            metrics = {'accuracy': tf.metrics.accuracy(y_copy, pred)}

        with tf.variable_scope('loss'):
            tile_eye_w, AA_T_w = self._tile_eye(A_w)
            tile_eye_w = tf.reshape(tile_eye_w, [-1, int(AA_T_w.shape[1]), int(AA_T_w.shape[1])])
            AA_T_w_sub_I = tf.subtract(AA_T_w, tile_eye_w)
            penalized_term_w = tf.square(tf.norm(AA_T_w_sub_I, axis=[-2, -1]))

            tile_eye_c, AA_T_c = self._tile_eye(A_c)
            tile_eye_c = tf.reshape(tile_eye_c, [-1, int(AA_T_c.shape[1]), int(AA_T_c.shape[1])])
            AA_T_c_sub_I = tf.subtract(AA_T_c, tile_eye_c)
            penalized_term_c = tf.square(tf.norm(AA_T_c_sub_I, axis=[-2, -1]))

            '''保证PREDICT mode下没有input_y的相关op'''
            input_y = tf.one_hot(input_y, depth=config.num_classes, dtype=tf.float32)
            '''这里发现softmax损失函数不收敛，原因不明'''
            input_y = self._label_smoothing(input_y, epsilon=config.smoothing_rate)
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=input_y)
            # loss = cross_entropy + penalized_term_w*config.penal_parms + penalized_term_c*config.penal_parms
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=input_y)
            loss = tf.reduce_mean(cross_entropy,
                                  1) + penalized_term_w * config.penal_parms + penalized_term_c * config.penal_parms
            loss = tf.reduce_mean(loss)
        if mode == tf.estimator.ModeKeys.EVAL:
            return logits, loss, pred, metrics

        with tf.variable_scope('optimizer'):
            optim_func = tf.train.AdamOptimizer(config.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.clip_by_norm)
            train_op = optim_func.apply_gradients(zip(grads, tvars), global_step=tf.train.get_global_step())

        return train_op, logits, loss, pred, metrics


def model_fn(features, labels, mode, params):
    x= features # dict
    if mode == tf.estimator.ModeKeys.PREDICT:
        softmax_score = mymodel(x, labels, mode,1.0,1.0)
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=softmax_score
                                          )
        return spec
    else:
        if mode == tf.estimator.ModeKeys.EVAL:
            _, loss, _ , metrics = mymodel(x,labels, mode,1.0,1.0)
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops = metrics
            )
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op, _, loss, _, metrics = mymodel(x,labels, mode, params['input_keep_prob'], params['dropout_keep_prob'])
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops = metrics
            )
        return spec


def main():
    feature_spec = {'sequence': tf.FixedLenSequenceFeature(dtype=tf.int64, shape=[], allow_missing=True),
                    'chars': tf.FixedLenSequenceFeature(dtype=tf.int64, shape=[], allow_missing=True)}

    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example."""
        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                               shape=[None],
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    print('mode:', config.mode)
    '''train mode dataset'''
    dataset = Dataset(config)
    input_fn = lambda: dataset._batched_data(path=config.tf_record_train)
    input_eval = lambda: dataset._batched_data(path=config.tf_record_eval)
    input_test = lambda: dataset._batched_data(path=config.tf_record_test)
    '''分布式策略'''
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    params = {'input_keep_prob': config.input_keep_prob,
              'dropout_keep_prob': config.dropout_keep_prob}
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=4)
    '''
    session_config = tf.estimator.RunConfig(train_distribute=distribution,
                                       save_checkpoints_steps=500,
                                     session_config=session_config)
    '''
    session_config = tf.estimator.RunConfig(train_distribute=distribution,
                                            eval_distribute=None,
                                            save_checkpoints_steps=250,
                                            session_config=session_config)

    if not os.path.join(config.base_dir, 'mymodeldir'):
        os.makedirs(os.path.join(config.base_dir, 'mymodeldir'))
    est_model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=os.path.join(config.base_dir, 'mymodeldir'),
                                       config=session_config, params=params)

    early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        est_model,
        metric_name='loss',
        max_steps_without_decrease=5000,
        min_steps=100,
        run_every_steps=500,
        run_every_secs=None)

    exporter = tf.estimator.BestExporter(
        name="best_exporter_2channel",
        serving_input_receiver_fn=serving_input_receiver_fn,
        exports_to_keep=5)

    tf.estimator.train_and_evaluate(
        est_model,
        train_spec=tf.estimator.TrainSpec(input_fn, hooks=[early_stopping], max_steps=3000),
        eval_spec=tf.estimator.EvalSpec(input_eval, throttle_secs=10, exporters=exporter, steps=None))

if __name__ == "__main__":
    config = Config()
    mymodel = MyModel()
    main()


