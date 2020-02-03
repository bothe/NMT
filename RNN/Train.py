from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from Checkpoint import load_checkpoint
from Dataset import load_word_dic, get_dataset, id_to_word
from Model import Encoder, Decoder
from Utils import korea_time, check_valid_path, create_folder


class Trainer():
    def __init__(self, **kwargs):
        dataset_folder = Path(kwargs["dataset_folder"]).resolve()
        check_valid_path(dataset_folder)
        result_folder = kwargs["result_folder"]

        self.initial_epoch = 1
        self.test_mode = kwargs["test"]
        self.epochs = kwargs["epochs"]
        self.use_label_smoothing = kwargs["label_smoothing"]

        self.ckpt_path = kwargs["ckpt_path"]
        self.ckpt_epoch = kwargs["ckpt_epoch"]

        # model에 필요한 폴더 및 파일 생성
        self.log_folder, self.ckpt_folder, self.image_folder = create_folder(result_folder)
        if not self.test_mode:
            self.training_result_file = self.log_folder / "training_result.txt"
        self.test_result_file = None

        # kwargs 값 저장
        msg = ""
        for k, v in list(kwargs.items()):
            msg += "{} = {}\n".format(k, v)
        msg += "new model checkpoint path = {}\n".format(self.ckpt_folder)
        with (self.log_folder / "model_settings.txt").open("w", encoding="utf-8") as fp:
            fp.write(msg)

        # 필요한 data를 불러옴
        self.src_word2id, self.src_id2word, self.src_vocab_size = load_word_dic(dataset_folder / "src_word2id.pkl")
        self.tar_word2id, self.tar_id2word, self.tar_vocab_size = load_word_dic(dataset_folder / "tar_word2id.pkl")

        if not self.test_mode:
            train_src, num_train_src = get_dataset(self.src_word2id, dataset_folder / "train_src.txt", False, True,
                                                   True)
            train_tar, num_train_tar = get_dataset(self.tar_word2id, dataset_folder / "train_tar.txt", True, True, True)
            if num_train_src != num_train_tar:
                raise Exception("source 데이터셋({})과 target 데이터셋({})의 크기가 다릅니다.".format(
                    num_train_src, num_train_tar))

            self.num_train = num_train_src
            self.train_dataset = tf.data.Dataset.from_generator(lambda: zip(train_src, train_tar), (tf.int32, tf.int32))
            self.train_dataset = self.train_dataset.cache().shuffle(self.num_train + 1).padded_batch(
                batch_size=kwargs["batch_size"], padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None])),
                padding_values=(self.src_word2id["<PAD>"], self.tar_word2id["<PAD>"])).prefetch(1)

        test_src_path = dataset_folder / "test.txt"
        if test_src_path.exists():
            test_src, self.num_test = get_dataset(self.src_word2id, test_src_path, False, True, False)

            # self.test_src_max_len = max([len(sentence) for sentence in test_src])
            # padded_test_src = tf.keras.preprocessing.sequence.pad_sequences(
            #    test_src, maxlen = self.test_src_max_len, padding = 'post',
            #    dtype = 'int32', value = self.src_word2id["<PAD>"])

            self.test_dataset = tf.data.Dataset.from_generator(lambda: test_src, tf.int32)
            self.test_dataset = self.test_dataset.cache().batch(1).prefetch(1)
            self.test_result_file = self.log_folder / "test_result.txt"

        elif self.test_mode:
            raise FileNotFoundError("[ {} ] 경로가 존재하지 않습니다.".format(test_src_path))

        self.encoder = Encoder(self.src_vocab_size, kwargs["embedding_size"], kwargs["hidden_size"],
                               kwargs["dropout_rate"],
                               kwargs["gru"], kwargs["bi"])
        self.decoder = Decoder(self.tar_vocab_size, kwargs["embedding_size"], kwargs["hidden_size"],
                               kwargs["attention_size"],
                               kwargs["dropout_rate"], kwargs["gru"], kwargs["bi"])

        # 아래 line 6줄은 colab에서 한글 깨짐을 방지하기 위한 부분으로 생략해도 됩니다.
        # %config InlineBackend.figure_format = 'retina'
        # !apt -qq -y install fonts-nanum
        fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
        font = fm.FontProperties(fname=fontpath, size=9)
        plt.rc('font', family='NanumBarunGothic')
        mpl.font_manager._rebuild()

    def start(self):
        if self.test_mode:
            self.test()
        else:
            self.train()

    def train(self):
        self.optimizer = tf.keras.optimizers.Adam()
        if self.use_label_smoothing:
            self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        else:
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_metric = tf.keras.metrics.Mean(name="train_loss")

        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder, opt=self.optimizer)

        if self.ckpt_path is not None:
            fname, self.initial_epoch = load_checkpoint(Path(self.ckpt_path).resolve(), self.ckpt_epoch)
            print("\nCheckpoint File : {}\n".format(fname))
            ckpt.mapped = {"encoder": self.encoder, "decoder": self.decoder, "opt": self.optimizer}
            ckpt.restore(fname)

        progbar = tf.keras.utils.Progbar(target=self.num_train)

        self.count = 0
        for epoch in range(self.initial_epoch, self.initial_epoch + self.epochs):
            progbar.update(0)
            self.loss_metric.reset_states()

            start_time = korea_time(None)
            for train_src, train_tar in self.train_dataset:
                num_data = K.int_shape(train_src)[0]
                self.forward(train_src, train_tar)

                progbar.add(num_data)

            end_time = korea_time(None)

            epoch_loss = self.loss_metric.result()

            ckpt_prefix = self.ckpt_folder / "Epoch-{}_Loss-{:.5f}".format(epoch, epoch_loss)
            ckpt.save(file_prefix=ckpt_prefix)

            print("Epoch = [{:5d}]    Loss = [{:8.6f}]\n".format(epoch, epoch_loss))

            # model result 저장
            msg = "Epoch = [{:5d}] - End Time [ {} ]\n".format(epoch, end_time.strftime("%Y/%m/%d %H:%M:%S"))
            msg += "Elapsed Time = {}\n".format(end_time - start_time)
            msg += "Loss : [{:8.6f}]\n".format(epoch_loss)
            msg += " - " * 15 + "\n\n"

            with self.training_result_file.open("a+", encoding="utf-8") as fp:
                fp.write(msg)

            if self.test_result_file is not None:
                self.translate(epoch)

    def get_loss(self, labels, logits):
        # labels shape : (sequence_length, 1, )
        # logits shape : (sequence_length, vocab_size, )

        # decoder에서 pad 부분은 loss에서 제외함
        loss_masking = tf.math.not_equal(labels, self.tar_word2id["<PAD>"])
        if self.use_label_smoothing:
            labels = K.one_hot(labels, self.tar_vocab_size)
            labels = self.label_smoothing(labels, self.tar_vocab_size)

        loss = self.loss_function(labels, logits)

        # loss_masking에는 True, False값으로 이루어져 있으므로 숫자로 바꿔줌
        loss_masking = tf.cast(loss_masking, loss.dtype)
        loss *= loss_masking

        # teacher forcing을 사용함으로 target의 길이는 항상 1
        return tf.reduce_mean(loss)

    def label_smoothing(self, inputs, vocab_size, epsilon=0.1):
        vocab_size = K.int_shape(inputs)[-1]
        return ((1 - epsilon) * inputs) + (epsilon / vocab_size)

    @tf.function(input_signature=[tf.TensorSpec((None, None), tf.int32), tf.TensorSpec((None, None), tf.int32)])
    def forward(self, train_src, train_tar):
        loss = 0.0
        batch_size = tf.shape(train_src)[0]

        with tf.GradientTape() as tape:
            enc_outputs, enc_hidden = self.encoder(train_src, True)
            dec_hidden = enc_hidden
            dec_inputs = tf.expand_dims(train_tar[:, 0], 1)  # (batch_size, 1)

            for idx in range(1, tf.shape(train_tar)[1]):  # sequence_length 만큼 반복
                logits, dec_hidden, _ = self.decoder(enc_outputs, dec_inputs, dec_hidden, True)
                loss += self.get_loss(train_tar[:, idx], logits)

                dec_inputs = tf.expand_dims(train_tar[:, idx], 1)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        loss_mean = loss / tf.cast(tf.shape(train_src)[1], tf.float32)  # sequence_length로 나눔
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        self.loss_metric.update_state(loss_mean)

    def test(self):
        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)
        fname, _ = load_checkpoint(self.ckpt_path, self.ckpt_epoch)
        print("\nCheckpoint File : {}\n".format(fname))

        # model만 불러옴
        ckpt.mapped = {"encoder": self.encoder, "decoder": self.decoder}
        ckpt.restore(fname).expect_partial()

        self.translate("Test")

    def translate(self, epoch):
        results = []
        att_weights_list = []

        for test_src in self.test_dataset:
            dec_inputs = tf.expand_dims([self.tar_word2id["<START>"]], 1)  # (1, 1)
            src_seq_len = K.int_shape(test_src)[1]
            tar_list = []
            att_graph = np.zeros((20, src_seq_len))  # 최대 20 글자까지 예측

            # 불필요한 반복 계산을 피하기 위해 encoder의 output은 미리 구함
            enc_outputs, enc_hidden = self.encoder(test_src, False)
            dec_hidden = enc_hidden

            for idx in range(20):
                # logits shape = [1, 1, vocab_size]
                # att_weights = [1, 1, attention_size]
                logits, dec_hidden, att_weights = self.decoder(enc_outputs, dec_inputs, dec_hidden, False)
                word_id = K.get_value(K.argmax(logits[0], axis=-1))[0]
                word = self.tar_id2word[word_id].split("/")[0]
                tar_list.append(word)

                # attention graph의 shape는 (target sequence, source sequence)이므로
                # att_weights의 shape를 (1, enc_sequence_len, 1)에서
                # (1, enc_sequence_len)로 바꿔줌.
                att_weights = tf.reshape(att_weights, (1, -1))
                att_graph[idx] = att_weights.numpy()

                if word == "<END>":
                    break

                dec_inputs = tf.expand_dims([word_id], 1)

            results.append(tar_list)
            att_weights_list.append(att_graph)

        self.save_results(results, att_weights_list, epoch)

    def save_results(self, tar_list, att_weights, epoch):
        image_epoch_folder = self.image_folder / "epoch-{}".format(epoch)
        image_epoch_folder.mkdir()

        dataset = zip(self.test_dataset, tar_list, att_weights)

        with self.test_result_file.open("a+", encoding="utf-8") as fp:
            fp.write("Epoch = [{:5d}]\n".format(epoch))

        for idx, (src_id, tar, weights) in enumerate(dataset):
            # <END> tag는 제외함
            src = id_to_word([K.get_value(num) for num in src_id[0][: -1]], self.src_id2word)

            src_sentence = [word.split("/")[0] for word in src]
            tar_sentence = [word.split("/")[0] for word in tar if word != "<END>"]

            with self.test_result_file.open("a+", encoding="utf-8") as fp:
                msg = "Source : {}\n".format(" ".join(src_sentence))
                if len(tar_sentence):
                    msg += "Target : {}\n\n".format(" ".join(tar_sentence))
                else:
                    msg += "Target : 번역 결과가 없습니다.\n\n"
                fp.write(msg)

            if len(tar_sentence):
                self.plot_attention(weights, src_sentence, tar_sentence, image_epoch_folder, idx + 1)

        with self.test_result_file.open("a+", encoding="utf-8") as fp:
            fp.write(" - - " * 10 + "\n\n")

    def plot_attention(self, att_weights, src_sentence, tar_sentence, image_epoch_folder, idx):
        sample_src = " ".join(src_sentence[: 5])

        # tar_sentence에서 마지막 token이 "<END>"라면 제외
        y_len = len(tar_sentence)
        if tar_sentence[-1] == "<END>":
            y_len -= 1

        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(1, 1, 1)
        graph = ax.matshow(att_weights[: y_len, :], cmap="viridis")
        fontdict = {"fontsize": 24}

        # x축 : 원문, y축 : 번역본
        ax.set_xticks(range(len(src_sentence) + 1))  # <END> tag를 포함
        ax.set_yticks(range(y_len))

        ax.set_xticklabels(src_sentence + ["<END>"], rotation=90, fontdict=fontdict)
        ax.set_yticklabels([token for token in tar_sentence if token != "<END>"], fontdict=fontdict)

        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        fig.colorbar(graph, cax=cax)

        plt.savefig(image_epoch_folder / "{}.png".format(sample_src))
        plt.close(fig)
