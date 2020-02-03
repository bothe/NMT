import tensorflow as tf
from tensorflow.keras import backend as K

class EmbeddingLayer(tf.keras.layers.Layer) :
    def __init__(self, vocab_size, embedding_size, dropout_rate) :
        super(EmbeddingLayer, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate) if dropout_rate else None
 
    def call(self, inputs, training = True) :
        # inputs shape : (batch_size, sequence_length)
        with tf.device("/cpu:0") :
            emb_x = self.embedding(inputs) # (batch_size, seqeucne_length, embedding_size)

        mask = tf.cast(tf.math.not_equal(inputs, 0), emb_x.dtype) # (batch_size, sequence_length)
        emb_x *= tf.expand_dims(mask, -1) # (batch_size, sequence_length, 1)

        if self.dropout_layer is not None:
            return self.dropout_layer(emb_x, training = training)

        return emb_x

class Encoder(tf.keras.Model) :
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout_rate, use_GRU, use_Bi) :
        super(Encoder, self).__init__()

        self.use_GRU = use_GRU
        self.use_Bi = use_Bi

        self.Embedding = EmbeddingLayer(vocab_size, embedding_size, dropout_rate)
        self.recurrent_layers = self._make_recurrent_layers(hidden_size, use_GRU, use_Bi)
        self.dropout_layers = [tf.keras.layers.Dropout(dropout_rate) if dropout_rate 
            else None for _ in range(len(hidden_size))
        ]

    def call(self, enc_inputs, training = True) :
        outputs = self.Embedding(enc_inputs, training = training)

        states_list = []
        for recurrent_layer, dropout_layer in zip(self.recurrent_layers, self.dropout_layers) :
            output_list = recurrent_layer(outputs) # initial_state를 따로 지정하지 않으면 0으로 채워짐
            outputs = output_list[0]

            if dropout_layer is not None :
                outputs = dropout_layer(outputs, training = training)

            states_list.append(self._get_states(output_list[1 : ]))

        return outputs, states_list

    def _make_recurrent_layers(self, hidden_sizes, use_GRU, use_Bi) :
        recurrent_layers = []
        for hidden_size in hidden_sizes :
            if use_GRU :
                layer = tf.keras.layers.GRU(hidden_size, return_sequences = True, return_state = True)
            else :
                layer = tf.keras.layers.LSTM(hidden_size, return_sequences = True, return_state = True)

            if use_Bi :
                layer = tf.keras.layers.Bidirectional(layer) # merge mode : concat

            recurrent_layers.append(layer)

        return recurrent_layers

    def _get_states(self, states_list) :
        if self.use_GRU :
            if self.use_Bi : 
                # [forward_hidden_state, backward_hidden_state]
                states = tf.keras.layers.Concatenate()(states_list)
            else : 
                # [hidden_state]
                states = states_list[0]
        else :
            if self.use_Bi : 
                # [forward_hidden_state, forward_cell_state, backward_hidden_state, backward_cell_state]
                hidden_states = tf.keras.layers.Concatenate()([states_list[0], states_list[2]])
                cell_states = tf.keras.layers.Concatenate()([states_list[1], states_list[3]])
                states = [hidden_states, cell_states]
            else :
                # [hidden_state, cell_state]
                states = states_list

        return states

class Decoder(tf.keras.Model) :
    def __init__(self, vocab_size, embedding_size, hidden_size, attention_size, dropout_rate, use_GRU, use_Bi) :
        super(Decoder, self).__init__()

        self.use_GRU = use_GRU
        self.num_layers = len(hidden_size)

        self.Embedding = EmbeddingLayer(vocab_size, embedding_size, dropout_rate)
        self.recurrent_layers = self._make_recurrent_layers(hidden_size, use_GRU, use_Bi)
        self.dropout_layers = [tf.keras.layers.Dropout(dropout_rate) if dropout_rate 
            else None for _ in range(self.num_layers)
        ]
        self.attention_layer = AttentionLayer(attention_size)
        self.linear = tf.keras.layers.Dense(vocab_size)

    def call(self, enc_outputs, dec_inputs, dec_hidden, training = True) :
        dec_outputs = self.Embedding(dec_inputs, training = training) # (sequence_length, embedding_size, )

        dec_hidden_for_attention = self._get_hidden_state_for_attention(dec_hidden)
        context_vector, att_weights = self.attention_layer(dec_hidden_for_attention, enc_outputs, enc_outputs)

        # (sequence_length, embedding_size + attention_size, )
        outputs = tf.keras.layers.Concatenate()([dec_outputs, context_vector])

        states_list = []
        for recurrent_layer, dropout_layer, states in zip(self.recurrent_layers, self.dropout_layers, dec_hidden) :
            output_list = recurrent_layer(outputs, initial_state = states)
            outputs = output_list[0]

            if dropout_layer is not None :
                outputs = dropout_layer(outputs, training = training)

            states_list.append(self._get_states(output_list[1 : ]))

        logits = self.linear(outputs)

        return logits, states_list, att_weights

    def _make_recurrent_layers(self, hidden_sizes, use_GRU, use_Bi) :
        recurrent_layers = []
        for hidden_size in hidden_sizes :
            if use_Bi : # encoder에서 hidden_state를 "concat"한 것을 고려
                hidden_size *= 2

            if use_GRU :
                layer = tf.keras.layers.GRU(hidden_size, return_sequences = True, return_state = True)
            else :
                layer = tf.keras.layers.LSTM(hidden_size, return_sequences = True, return_state = True)

            recurrent_layers.append(layer)

        return recurrent_layers

    def _get_hidden_state_for_attention(self, states_list) :
        if self.num_layers > 1 :
            dec_hidden_for_attention = states_list[-1]
        else :
            dec_hidden_for_attention = states_list[0]

        if not self.use_GRU :
            dec_hidden_for_attention = dec_hidden_for_attention[0]

        return dec_hidden_for_attention

    def _get_states(self, states_list) :
        if self.use_GRU :
            # [hidden_state]
            states = states_list[0]
        else :
            # [hidden_state, cell_state]
            states = states_list

        return states

class AttentionLayer(tf.keras.layers.Layer) :
    def __init__(self, attention_size) :
        super(AttentionLayer, self).__init__()

        self.query_project = tf.keras.layers.Dense(attention_size)
        self.key_project = tf.keras.layers.Dense(attention_size)
        self.attention_project = tf.keras.layers.Dense(1)

    def call(self, query, key, value) :
        # query = "decoder hidden state" (dec_hidden_size, )
        # key = value = "encoder output" (enc_sequence_length, enc_hidden_size, )
        expanded_query = tf.expand_dims(query, 1) # (1, dec_hidden_size, )
        query_proj = self.query_project(expanded_query)
        key_proj = self.key_project(key)
        sum_proj = query_proj + key_proj # (enc_sequence_length, attention_size, )
        
        act = tf.keras.activations.tanh(sum_proj)
        att_proj = self.attention_project(act)
        att_weights = tf.nn.softmax(att_proj, axis = 1) # (enc_sequence_length, 1, )

        # (1, hidden_size, )
        context_vector = tf.expand_dims(tf.reduce_sum(att_weights * value, axis = 1), 1)

        return context_vector, att_weights