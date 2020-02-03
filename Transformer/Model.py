import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

class Transformer(tf.keras.Model) :
    def __init__(self, source_vocab_size, target_vocab_size, padding_value, 
        num_layers, num_heads, embedding_size, hidden_size, dropout_rate, use_conv) :
        super(Transformer, self).__init__()

        self.encoder = Encoder(dropout_rate, num_layers, num_heads, hidden_size, 
                source_vocab_size, embedding_size, padding_value, use_conv)
        self.decoder = Decoder(dropout_rate, num_layers, num_heads, hidden_size, 
                target_vocab_size, embedding_size, use_conv)
        self.linear = tf.keras.layers.Dense(target_vocab_size)

    def call(self, enc_inputs, dec_inputs, training = True) :
        enc_outputs, padding_mask = self.encoder(enc_inputs, training)
        dec_outputs, att_weights = self.decoder(dec_inputs, enc_outputs, padding_mask, training)
        logits = self.linear(dec_outputs)

        return logits, att_weights

class PositionalEncodingLayer(tf.keras.layers.Layer) :
    def __init__(self, vocab_size, embedding_size, dropout_rate) :
        super(PositionalEncodingLayer, self).__init__()
        
        self.embedding_size = embedding_size
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = vocab_size, 
            output_dim = self.embedding_size)
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate) if dropout_rate else None
        self._PE = self._make_positional_encoding(10000)
    
    def call(self, inputs, training) :
        sequence_length = tf.shape(inputs)[1]
        with tf.device("/cpu:0") :
            emb_outputs = self.embedding_layer(inputs) # (sequence_length, embedding_size, )
        emb_outputs *= (self.embedding_size ** 0.5)
        outputs = emb_outputs + self._PE[:sequence_length, :]

        if self.dropout_layer is not None :
            outputs = self.dropout_layer(outputs, training = training)
            
        return outputs

    def _make_positional_encoding(self, sequence_length) :
        # positional encoding
        # PE.shape(2-D) : (sequence_length, embedding_size)
        PE = np.array(
            # i            => 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
            # 2 * (i // 2) => 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, ...
            [[outer_idx / pow(float(10000), (2 * (inner_idx // 2)) / self.embedding_size) for inner_idx in range(self.embedding_size)]
            for outer_idx in range(sequence_length)]
        )
        PE[:, 0 : : 2] = tf.math.sin(PE[:, 0 : : 2]) # 0 2 4 6 8 10 ...
        PE[:, 1 : : 2] = tf.math.cos(PE[:, 1 : : 2]) # 1 3 5 7 9 11 ...
        
        return tf.constant(PE, tf.float32)

class Encoder(tf.keras.layers.Layer) :
    def __init__(self, dropout_rate, num_layers, num_heads, hidden_size, 
                 vocab_size, embedding_size, padding_value, use_conv) :
        super(Encoder, self).__init__()
        
        self.padding_value = padding_value
        self.positional_encdoing_layer = PositionalEncodingLayer(vocab_size, embedding_size, dropout_rate)
        self.enc_blocks = [EncoderBlock(dropout_rate, hidden_size, num_heads, use_conv) for _ in range(num_layers)]

    def call(self, enc_inputs, training = True) :
        enc_seq_len = tf.shape(enc_inputs)[1]
        padding_mask = self._make_padding_mask(enc_inputs, enc_seq_len)

        outputs = self.positional_encdoing_layer(enc_inputs, training)
        for enc_block in self.enc_blocks :
            outputs = enc_block(outputs, padding_mask, training)
            
        # decoder에서 사용하기 위해 padding_mask를 반환
        return outputs, padding_mask

    def _make_padding_mask(self, inputs, sequence_length) :
        # output shape : (1, 1, decoder_seqeuence_lngth, )
        mask = tf.math.equal(inputs, self.padding_value)
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(tf.cast(mask, tf.float32), (-1, 1, 1, sequence_length))
        return mask

class EncoderBlock(tf.keras.layers.Layer) :
    def __init__(self, dropout_rate, hidden_size, num_heads, use_conv) :
        super(EncoderBlock, self).__init__()
        
        self.mha = MultiHeadAttentionLayer(num_heads, hidden_size)
        self.ffa = FeedForwardLayer(hidden_size, use_conv)
        
        if dropout_rate :
            self.dr1 = tf.keras.layers.Dropout(dropout_rate)
            self.dr2 = tf.keras.layers.Dropout(dropout_rate)
        else :
            self.dr1 = self.dr2 = None
        
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        
    def call(self, inputs, mask, training) :
        outputs, _ = self.mha(inputs, inputs, inputs, mask)
        if self.dr1 is not None :
            outputs = self.dr1(outputs, training = training)
        
        outputs = self.ln1(inputs + outputs)
        inputs = outputs
        
        outputs = self.ffa(inputs)
        if self.dr2 is not None :
            outputs = self.dr2(outputs, training = training)
        
        outputs = self.ln2(inputs + outputs)
        
        return outputs

class Decoder(tf.keras.layers.Layer) :
    def __init__(self, dropout_rate, num_layers, num_heads, hidden_size, 
                 vocab_size, embedding_size, use_conv) :
        super(Decoder, self).__init__()

        self.positional_encdoing_layer = PositionalEncodingLayer(vocab_size, embedding_size, dropout_rate)
        self.dec_blocks = [DecoderBlock(dropout_rate, hidden_size, num_heads, use_conv) for _ in range(num_layers)]

    def call(self, dec_inputs, enc_outputs, padding_mask, training = True) :
        dec_seq_len = tf.shape(dec_inputs)[1]
        look_ahead_mask = self._make_look_ahead_mask(dec_seq_len)

        dec_outputs = self.positional_encdoing_layer(dec_inputs, training)
        for dec_block in self.dec_blocks :
            dec_outputs, att_weights = dec_block(dec_outputs, enc_outputs, padding_mask, look_ahead_mask, training)
            
        return dec_outputs, att_weights

    def _make_look_ahead_mask(self, sequence_length) :
        # decoder에서 앞 단어만을 참조하기 위한 mask
        # output shape : (1, 1, sequence_length, sequence_length)

        # 0 1 1 1 1
        # 0 0 1 1 1
        # 0 0 0 1 1
        # 0 0 0 0 1
        # 0 0 0 0 0
        upperTri = 1.0 - tf.linalg.band_part(tf.ones([sequence_length, sequence_length]), -1, 0)
        upperTri = tf.reshape(upperTri, (1, 1, sequence_length, sequence_length))
        upperTri = tf.cast(upperTri, tf.float32)
        return upperTri

class DecoderBlock(tf.keras.layers.Layer) :
    def __init__(self, dropout_rate, hidden_size, num_heads, use_conv) :
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttentionLayer(num_heads, hidden_size) # self-attention
        self.mha2 = MultiHeadAttentionLayer(num_heads, hidden_size) # encoder-decoder attention
        self.ffa = FeedForwardLayer(hidden_size, use_conv)

        if dropout_rate :
            self.dr1 = tf.keras.layers.Dropout(dropout_rate)
            self.dr2 = tf.keras.layers.Dropout(dropout_rate)
            self.dr3 = tf.keras.layers.Dropout(dropout_rate)
        else :
            self.dr1 = self.dr2 = self.dr3 = None

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)

    def call(self, dec_outputs, enc_outputs, padding_mask, look_ahead_mask, training) :
        inputs = dec_outputs
        outputs, att_weights = self.mha1(dec_outputs, dec_outputs, dec_outputs, look_ahead_mask)
        if self.dr1 is not None :
            outputs = self.dr1(outputs, training = training)
        outputs = self.ln1(inputs + outputs)

        inputs = outputs
        outputs, att_weights = self.mha2(outputs, enc_outputs, enc_outputs, padding_mask) # q, k, v
        if self.dr2 is not None :
            outputs = self.dr2(outputs, training = training)
        outputs = self.ln2(inputs + outputs)

        inputs = outputs
        outputs = self.ffa(inputs)
        if self.dr3 is not None :
            outputs = self.dr2(outputs, training = training)
        outputs = self.ln3(inputs + outputs)

        return outputs, att_weights
    
class FeedForwardLayer(tf.keras.layers.Layer) :
    def __init__(self, hidden_size, use_conv = False) :
        super(FeedForwardLayer, self).__init__()
        
        if use_conv :
            self.fc1 = tf.keras.layers.Conv1D(hidden_size * 4, 1, activation = tf.keras.activations.relu)
            self.fc2 = tf.keras.layers.Conv1D(hidden_size, 1)
        else :
            self.fc1 = tf.keras.layers.Dense(hidden_size * 4, activation = tf.keras.activations.relu)
            self.fc2 = tf.keras.layers.Dense(hidden_size)
        
    def call(self, inputs) :
        outputs = self.fc1(inputs)
        return self.fc2(outputs)
                    
class MultiHeadAttentionLayer(tf.keras.layers.Layer) :
    def __init__(self, num_heads, hidden_size) :
        super(MultiHeadAttentionLayer, self).__init__()
        
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.query_projection = tf.keras.layers.Dense(self.hidden_size)
        self.key_projection = tf.keras.layers.Dense(self.hidden_size)
        self.value_projection = tf.keras.layers.Dense(self.hidden_size)
        self.output_projection = tf.keras.layers.Dense(self.hidden_size)
            
    def call(self, q, k, v, mask) :
        # q = decoder outputs
        # k = v = encoder outputs
        qp = self.query_projection(q)
        kp = self.key_projection(k)
        vp = self.value_projection(v)
        
        qs = self._split_heads(qp)
        ks = self._split_heads(kp)
        vs = self._split_heads(vp)
        
        outputs, att_weights = self._scaled_dot_product_attention(qs, ks, vs, mask)
        outputs = self._concat_heads(outputs)
        outputs = self.output_projection(outputs)
        
        return outputs, att_weights

    def _scaled_dot_product_attention(self, qs, ks, vs, mask) :
        # padding_mask : adding 부분은 제외해야 함.
        # look_ahead_mask : decoder에서 self-attention을 구할 때
        # 기준 단어보다 뒤에 있는 단어들은 해당 단어에 영향을 주면 안됨.
        
        # matmul
        # output : (num_heads, query_sequence_length, key_sequence_length, )
        qk_matmul = tf.matmul(qs, ks, transpose_b = True)
        
        # scale
        qk_scale = qk_matmul / tf.math.sqrt(tf.cast(tf.shape(ks)[-1], tf.float32))
        
        # masking
        if mask is not None :
            mask *= -1e9
            qk_scale += mask

        # softmax
        qk_att = tf.nn.softmax(qk_scale, axis = -1)
        
        # matmul
        # output : (num_heads, query_sequence_lenth, depth, )
        qv_matmul = tf.matmul(qk_att, vs)
        
        return qv_matmul, qk_att
    
    def _split_heads(self, inputs) :
        # input shape : (sequence_length, hidden_size, )
        # output shape : (num_heads, sequence_length, hidden_size // num_heads, )
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        sequence_length = inputs_shape[1]
        
        # 각 head마다 head_size만큼을 할당
        head_size = self.hidden_size // self.num_heads
        outputs = tf.reshape(inputs, [batch_size, sequence_length, self.num_heads, head_size])
        
        # (sequence_length, num_heads, head_size, )
        # => (num_heads, sequence_length, head_size, )
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        
        return outputs
    
    def _concat_heads(self, inputs) :
        # input shape : (num_heads, sequence_length, hidden_size // num_heads, )
        # output shape : (sequence_length, hidden_size, )
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        sequence_length = inputs_shape[2]
        
        # (num_heads, sequence_length, hidden_size // num_heads, )
        # => (sequence_length, num_heads, hidden_size // num_heads, )
        outputs = tf.transpose(inputs, [0, 2, 1, 3])
        
        # (sequence_length, hidden_size, )
        return tf.reshape(outputs, [batch_size, sequence_length, self.hidden_size])