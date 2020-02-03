from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

from Checkpoint import load_checkpoint
from Dataset import load_word_dic, get_dataset, id_to_word
from Model import Transformer
from Utils import korea_time, check_valid_path, create_folder


class Trainer():
    def __init__(self, **kwargs):
        dataset_folder = Path(kwargs["dataset_folder"]).resolve()
        check_valid_path(dataset_folder)
        result_folder = kwargs["result_folder"]

        self.initial_epoch = 1
        self.test_mode = kwargs["test"]
        self.epochs = kwargs["epochs"]
        self.hidden_size = kwargs["hidden_size"]
        self.num_heads = kwargs["heads"]
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
            # encoder data : <END> tag 추가
            # decoder data : 1) input = <START> tag만 추가 2)output = <END> tag만 추가
            train_src, num_train_src = get_dataset(self.src_word2id, dataset_folder / "train_src.txt", False, True,
                                                   True)
            train_tar, num_train_tar = get_dataset(self.tar_word2id, dataset_folder / "train_tar.txt", True, True, True)
            if num_train_src != num_train_tar:
                raise Exception("한글 데이터셋({})과 영어 데이터셋({})의 크기가 다릅니다.".format(
                    num_train_src, num_train_tar))

            self.num_train = num_train_src
            self.train_dataset = tf.data.Dataset.from_generator(lambda: zip(train_src, train_tar), (tf.int32, tf.int32))
            self.train_dataset = self.train_dataset.cache().shuffle(self.num_train + 1).padded_batch(
                batch_size=kwargs["batch_size"], padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None])),
                padding_values=(self.src_word2id["<PAD>"], self.tar_word2id["<PAD>"])).prefetch(1)

        test_src_path = dataset_folder / "test.txt"
        if test_src_path.exists():
            test_src, self.num_test = get_dataset(self.src_word2id, test_src_path, False, True, False)
            self.test_dataset = tf.data.Dataset.from_generator(lambda: test_src, tf.int32)
            self.test_dataset = self.test_dataset.cache().batch(1).prefetch(1)
            self.test_result_file = self.log_folder / "test_result.txt"
        elif self.test_mode:
            raise FileNotFoundError("[ {} ] 경로가 존재하지 않습니다.".format(test_src_path))

        self.transformer = Transformer(self.src_vocab_size, self.tar_vocab_size, self.src_word2id["<PAD>"],
                                       kwargs["num_layers"], kwargs["heads"], kwargs["embedding_size"],
                                       kwargs["hidden_size"],
                                       kwargs["dropout_rate"], kwargs["use_conv"])

    def start(self):
        if self.test_mode:
            self.test()
        else:
            self.train()

    def train(self):
        self.optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        if self.use_label_smoothing:
            self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        else:
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")

        ckpt = tf.train.Checkpoint(model=self.transformer, opt=self.optimizer)

        if self.ckpt_path is not None:
            fname, self.initial_epoch = load_checkpoint(Path(self.ckpt_path).resolve(), self.ckpt_epoch)
            print("\nCheckpoint File : {}\n".format(fname))
            ckpt.mapped = {"model": self.transformer, "opt": self.optimizer}
            ckpt.restore(fname)

        progbar = tf.keras.utils.Progbar(target=self.num_train)

        self.count = 0
        for epoch in range(self.initial_epoch, self.initial_epoch + self.epochs):
            K.set_value(self.optimizer.lr, self._get_lr(epoch))
            progbar.update(0)
            self.loss_metric.reset_states()
            self.acc_metric.reset_states()

            start_time = korea_time(None)
            for train_src, train_tar in self.train_dataset:
                num_data = K.int_shape(train_src)[0]
                logits = self.forward(train_src, train_tar)

                progbar.add(num_data)

            end_time = korea_time(None)

            epoch_loss = self.loss_metric.result()
            epoch_acc = self.acc_metric.result()

            ckpt_prefix = self.ckpt_folder / "Epoch-{}_Loss-{:.5f}_Acc-{:5f}".format(
                epoch, epoch_loss, epoch_acc)
            ckpt.save(file_prefix=ckpt_prefix)

            print("Epoch = [{:5d}]    Loss = [{:8.6f}]    Acc = [{:8.6f}]   LR = [{:.10f}]\n".format(
                epoch, epoch_loss, epoch_acc, K.get_value(self.optimizer.lr)))

            # model result 저장
            msg = "Epoch = [{:5d}] - End Time [ {} ]\n".format(epoch, end_time.strftime("%Y/%m/%d %H:%M:%S"))
            msg += "Elapsed Time = {}\n".format(end_time - start_time)
            msg += "Learning Rate = [{:.10f}]\n".format(K.get_value(self.optimizer.lr))
            msg += "Loss : [{:8.6f}] - Acc : [{:8.6f}]\n".format(epoch_loss, epoch_acc)
            msg += " - " * 15 + "\n\n"

            with self.training_result_file.open("a+", encoding="utf-8") as fp:
                fp.write(msg)

            if self.test_result_file is not None:
                self.translate(epoch)

    def test(self):
        ckpt = tf.train.Checkpoint(model=self.transformer)
        fname, _ = load_checkpoint(Path(self.ckpt_path).resolve(), self.ckpt_epoch)
        print("\nCheckpoint File : {}\n".format(fname))

        # model만 불러옴
        ckpt.mapped = {"model": self.transformer}
        ckpt.restore(fname).expect_partial()

        self.translate("Test")

    def _get_lr(self, step):
        return pow(self.hidden_size, -0.5) * min(pow(step, -0.5), step * pow(4000, -1.5))

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

        # loss를 구할 때 PAD 부분은 제외함
        return tf.reduce_sum(loss) / (tf.reduce_sum(loss_masking) + 1e-9)

    def label_smoothing(self, inputs, vocab_size, epsilon=0.1):
        vocab_size = K.int_shape(inputs)[-1]
        return ((1 - epsilon) * inputs) + (epsilon / vocab_size)

    @tf.function(input_signature=[tf.TensorSpec((None, None), tf.int32), tf.TensorSpec((None, None), tf.int32)])
    def forward(self, train_src, train_tar):
        # train_tar = <START> Token Token Token ... Token <END>
        # input_tar = <START> Token Token Token ... Token
        # output_tar = Token Token Token ... Token <END>
        enc_inputs = train_src
        dec_inputs = train_tar[:, : -1]
        dec_outputs = train_tar[:, 1:]

        with tf.GradientTape() as tape:
            logits, _ = self.transformer(enc_inputs, dec_inputs, True)
            loss = self.get_loss(dec_outputs, logits)

        grads = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.transformer.trainable_variables))

        self.loss_metric.update_state(loss)
        self.acc_metric.update_state(dec_outputs, logits)

        return logits

    def translate(self, epoch):
        results = []
        att_weights_list = []

        for test_src in self.test_dataset:
            dec_inputs = tf.expand_dims([self.tar_word2id["<START>"]], 1)  # (1, 1)
            tar_list = []

            # 불필요한 반복 계산을 피하기 위해 encoder의 output은 미리 구함
            enc_outputs, padding_mask = self.transformer.encoder(test_src, False)

            for idx in range(20):  # 최대 20글자까지 예측
                # shape : (1, sequence_length, vocab_size)
                # att_weights에는 kwargs["heads"] 만큼의 head개 저장되어 있음
                # att_weights shape : (1, num_heads, target_sentence_len, source_sentence_len)
                dec_outputs, att_weights = self.transformer.decoder(dec_inputs, enc_outputs, padding_mask, False)
                logits = self.transformer.linear(dec_outputs)

                # 마지막 word에 대한 logits만 선택. shape : (1, vocab_size)
                last_word_logits = logits[:, -1, :]

                word_id = K.get_value(K.argmax(last_word_logits, axis=-1))[0]
                word = self.tar_id2word[word_id].split("/")[0]
                tar_list.append(word)

                if word == "<END>":
                    break

                dec_inputs = tf.concat([dec_inputs, [[word_id]]], axis=-1)  # shape : (1, n)

            results.append(tar_list)
            att_weights_list.append(att_weights[0])  # batch_size 부분은 제거

        self.save_results(results, att_weights_list, epoch)

    def save_results(self, tar_list, att_weights, epoch):
        image_epoch_folder = self.image_folder / "epoch-{}".format(epoch)
        image_epoch_folder.mkdir()

        dataset = zip(self.test_dataset, tar_list, att_weights)

        with self.test_result_file.open("a+", encoding="utf-8") as fp:
            if isinstance(epoch, int):
                fp.write("Epoch = [{:5d}]\n".format(epoch))
            else:
                fp.write("Epoch = {}\n".format(epoch))

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
                    msg += "Target : 번역결과가 없습니다.\n\n"
                fp.write(msg)

            if len(tar_sentence):
                self.plot_attention(weights, src_sentence, tar_sentence, image_epoch_folder, idx + 1)

        with self.test_result_file.open("a+", encoding="utf-8") as fp:
            fp.write(" - - " * 10 + "\n\n")

    def plot_attention(self, att_weights, src_sentence, tar_sentence, image_epoch_folder, idx):
        sample_src = " ".join(src_sentence[: 5])
        save_folder = image_epoch_folder / sample_src
        if not save_folder.exists():
            save_folder.mkdir()

        # tar_sentence에서 마지막 token이 "<END>"라면 제외
        y_len = len(tar_sentence)
        if tar_sentence[-1] == "<END>":
            y_len -= 1

        # kwargs["heads"] 만큼의 attention graph를 작성
        for head_idx in range(self.num_heads):
            fig = plt.figure(figsize=(16, 16))
            ax = fig.add_subplot(1, 1, 1)
            graph = ax.matshow(att_weights[head_idx][: y_len, :], cmap="viridis")
            fontdict = {"fontsize": 24}

            # x축 : 한글, y축 : 영어
            ax.set_xticks(range(len(src_sentence) + 1))  # <END> tag를 포함
            ax.set_yticks(range(y_len))

            ax.set_xticklabels(src_sentence + ["<END>"], rotation=90, fontdict=fontdict)
            ax.set_yticklabels([token for token in tar_sentence if token != "<END>"], fontdict=fontdict)

            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
            fig.colorbar(graph, cax=cax)

            plt.savefig(save_folder / "Head-{}.png".format(head_idx + 1))
            plt.close(fig)
