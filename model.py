import os
import sys
import os.path
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from collections import Counter
from tensorflow.python.ops.rnn_cell import RNNCell


def _read_nodes(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if ':' in line:
                line = line.split(':')[0]
            data.extend(line.replace('\n', '').split(','))
        return data

def read_graph(filename, node_to_id):
    N = len(node_to_id)
    A = np.zeros((N,N), dtype=np.float32)
    with open(filename, 'r') as f:
        for line in f:
            edge = line.strip().split(',')
            if edge[0] in node_to_id and edge[1] in node_to_id:
                source_id = node_to_id[edge[0]]
                target_id = node_to_id[edge[1]]
                if len(edge) >= 3:
                    A[source_id,target_id] = float(edge[2])
                else:
                    A[source_id,target_id] = 1.0
    return A

def _build_vocab(filename):
    data = _read_nodes(filename)

    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    nodes, _ = list(zip(*count_pairs))
    nodes = list(nodes)
    nodes.insert(0,'-1') # index for mask
    node_to_id = dict(zip(nodes, range(len(nodes))))
    print(node_to_id)

    return nodes, node_to_id

def _file_to_node_ids(filename, node_to_id):
    data = []
    len_list = []
    with open(filename, 'r') as f:
        for line in f:
            if ':' in line:
                line = line.split(':')[0]
            seq = line.strip().split(',')
            ix_seq = [node_to_id[x] for x in seq if x in node_to_id]
            if len(ix_seq)>=2:
                data.append(ix_seq)
                len_list.append(len(ix_seq)-1)
    size = len(data)
    total_num = np.sum(len_list)
    return (data, len_list, size, total_num)

def to_nodes(seq, nodes):
    return list(map(lambda x: nodes[x], seq))

def read_raw_data(data_path=None):
    train_path = data_path + '-train.txt'
    valid_path = data_path +  '-val.txt'
    test_path = data_path +  '-test.txt'

    nodes, node_to_id = _build_vocab(train_path)
    train_data = _file_to_node_ids(train_path, node_to_id)
    valid_data = _file_to_node_ids(valid_path, node_to_id)
    test_data = _file_to_node_ids(test_path, node_to_id)
    print('Node Num:' + str(len(nodes)-1)) # Exclude the masking index 0
    print('train size:' + str(len(train_data[0])) + '; ' + 'test size:' + str(len(test_data[0])))
    return train_data, valid_data,  test_data,  nodes, node_to_id

def batch_generator(train_data, batch_size=50):
    x = []
    y = []
    xs = []
    ys = []
    ss = []
    train_seq = train_data[0]
    train_steps = train_data[1]
    batch_len = len(train_seq) // batch_size

    for i in range(batch_len):
        batch_steps = np.array(train_steps[i * batch_size : (i + 1) * batch_size])
        max_batch_steps = batch_steps.max()
        for j in range(batch_size):
            seq = train_seq[i * batch_size + j]
            padded_seq = np.pad(np.array(seq),(0, max_batch_steps-len(seq)+1),'constant') # padding with 0
            x.append(padded_seq[:-1])
            y.append(padded_seq[1:])
        x = np.array(x)
        y = np.array(y)
        xs.append(x)
        ys.append(y)
        ss.append(batch_steps)
        x = []
        y = []
    rest_len = len(train_steps[batch_len * batch_size : ])
    if rest_len != 0:
        batch_steps = np.array(train_steps[batch_len * batch_size : ])
        max_batch_steps = batch_steps.max()
        for j in range(rest_len):
            seq = train_seq[batch_len * batch_size + j]
            padded_seq = np.pad(np.array(seq),(0, max_batch_steps-len(seq)+1),'constant')
            x.append(padded_seq[:-1])
            y.append(padded_seq[1:])
        x = np.array(x)
        y = np.array(y)
        xs.append(x)
        ys.append(y)
        ss.append(batch_steps)
    # Enumerator over the batches.
    return xs, ys, ss


class SnidsaCell(RNNCell):
    """ Recurrent Unit Cell for SNIDSA."""

    def __init__(self, num_units, feat_in_matrix, activation=None, reuse=None):
        self._num_units = num_units
        self._num_nodes = int(feat_in_matrix.shape[0])
        self._feat_in_matrix = feat_in_matrix
        self._feat_in = int(feat_in_matrix.shape[1])
        self._activation = activation or tf.tanh

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        X = inputs[0]
        A = inputs[1]
        feat_in = self._feat_in
        feat_out = self._num_units
        num_nodes = self._num_nodes
        feat_in_matrix = self._feat_in_matrix

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Attentions"):
                struc, x = struc_atten(X, feat_in_matrix, A, feat_in)
            with tf.variable_scope("c_inputs"):
                xs_c = linear([x, struc], feat_out, False)
            with tf.variable_scope("h_inputs"):
                xs_h = linear([x, struc], feat_out, False)
            with tf.variable_scope("Gate"):
                concat = tf.sigmoid(
                    linear([X, struc], 2 * feat_out, True))
                if tf.__version__ == "0.12.1":
                    f, r = tf.split(1, 2, concat)
                else:
                    f, r = tf.split(axis=1, num_or_size_splits=2, value=concat)

            c = f * state + (1 - f) * xs_c

            # highway connection
            h = r * self._activation(c) + (1 - r) * xs_h

        return h, c


def struc_atten(X, feat_in_matrix, A, feat_out):
    batch_size = tf.shape(X)[0]
    num_nodes = tf.shape(feat_in_matrix)[0]
    with tf.variable_scope("linear_transf"):
        linear_transf_X = linear([X], feat_out, False)
        tf.get_variable_scope().reuse_variables()
        linear_transf_G = linear([feat_in_matrix], feat_out, False)
    with tf.variable_scope("strcuture_attention"):
        Wa = tf.get_variable("Wa", [2*feat_out, 1], dtype=tf.float32,
                           initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        # Repeat feature vectors of input: [[1], [2]] becomes [[1], [1], [2], [2]]
        repeated = tf.reshape(tf.tile(linear_transf_X, (1, num_nodes)), (batch_size * num_nodes, feat_out))  # (BN x F')
        # Tile feature vectors of full graph: [[1], [2]] becomes [[1], [2], [1], [2]]
        tiled = tf.tile(linear_transf_G, (batch_size, 1))  # (BN x F')
        # Build combinations
        combinations = tf.concat([repeated, tiled],1)  # (BN x 2F')
        combination_slices = tf.reshape(combinations, (batch_size, -1, 2 * feat_out))  # (B x N x 2F')

        dense = tf.squeeze(tf.contrib.keras.backend.dot(combination_slices, Wa), -1)  
        # Convert to B X N
        comparison = tf.equal(A, tf.constant(0, dtype=tf.float32))
        mask = tf.where(comparison, tf.ones_like(A) * -10e9, tf.zeros_like(A))
        masked = dense + mask

        struc_att = tf.nn.softmax(masked)  # (B x N)
        struc_att = tf.nn.dropout(struc_att, 1)  # Apply dropout to normalized attention coefficients (B x N)

        # Linear combination with neighbors' features
        struc = tf.matmul(struc_att, linear_transf_G)  # (B x F')

        struc = tf.nn.elu(struc)

    return struc, linear_transf_X


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term


class SNIDSA(object):
    def __init__(self, config, A, is_training=True):

        self.num_nodes = config.num_nodes
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        # self.model = config.model

        self.learning_rate = config.learning_rate
        self.dropout = config.dropout

        self._A = tf.constant(A, dtype=tf.float32, name="adjacency_matrix")

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(
                "embedding", [self.num_nodes,
                    self.embedding_dim], dtype=tf.float32)

        self.placeholders()
        self.loss_mask()
        self.graph_information()
        self.recurrent_layer()
        self.cost()
        self.optimize()

    def placeholders(self):
        self.batch_size = tf.placeholder(tf.int32, None)
        self._inputs = tf.placeholder(tf.int32, [None, None]) # [batch_size, num_steps]
        self._targets = tf.placeholder(tf.int32, [None, None])
        self._seqlen = tf.placeholder(tf.int32, [None])
        self.num_steps = tf.placeholder(tf.int32, None)

    def loss_mask(self):
        self._target_mask = tf.sequence_mask(self._seqlen, dtype=tf.float32)

    def graph_information(self):
        _neighbors = tf.nn.embedding_lookup(self._A, self._inputs)
        return _neighbors

    def input_embedding(self):
        _inputs = tf.nn.embedding_lookup(self.embedding, self._inputs)
        return _inputs

    def recurrent_layer(self):
        def creat_cell():
            cell = SnidsaCell(self.hidden_dim, self.embedding)
            if self.dropout < 1:
                return tf.contrib.rnn.DropoutWrapper(cell,
                    output_keep_prob=self.dropout)
            else:
                return cell

        cells = [creat_cell() for _ in range(self.num_layers)]
        cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        emb_inputs = self.input_embedding()
        _neighbors = self.graph_information()
        _outputs, _ = tf.nn.dynamic_rnn(cell=cell,
            inputs=(emb_inputs,_neighbors), sequence_length=self._seqlen, dtype=tf.float32)

        output = tf.reshape(tf.concat(_outputs, 1), [-1, self.hidden_dim])
        softmax_w = tf.get_variable(
            "softmax_w", [self.hidden_dim, self.num_nodes], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.num_nodes], dtype=tf.float32)
        self.flat_logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
         # Reshape logits to be a 3-D tensor for sequence loss
        self._logits = tf.reshape(self.flat_logits, [self.batch_size, self.num_steps, self.num_nodes])

    def cost(self):
        crossent = tf.contrib.seq2seq.sequence_loss(
            self._logits,
            self._targets,
            self._target_mask,
            average_across_timesteps=False,
            average_across_batch=False)
        loss = tf.reduce_sum(crossent, axis=[0])
        batch_avg = tf.reduce_sum(self._target_mask, axis=[0])
        batch_avg += 1e-12  # to avoid division by 0 for all-0 weights
        loss /= batch_avg
        # Update the cost
        self.cost = tf.reduce_sum(loss)
        # Calculate negative log-likelihood
        self.nll = tf.reduce_sum(crossent, axis = [1])

        pred = tf.nn.softmax(self.flat_logits)
        self.pred = tf.reshape(pred, [self.batch_size, self.num_steps, self.num_nodes])

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optim = optimizer.minimize(self.cost)


class Config(object):
    """Configuration of model"""
    num_layers = 1
    batch_size = 16
    embedding_dim = 32
    hidden_dim = 64
    num_epochs = 200
    valid_freq = 5
    patience = int(10/valid_freq) + 1
    model = 'snidsa'
    gpu_no = '0'
    #data_name = 'data/weibo-cascades'
    data_name = '../urban_project/twitter'
    learning_rate = 0.001
    dropout = 0.6
    random_seed = 1402

class Input(object):
    def __init__(self, config, data):
        self.batch_size = config.batch_size
        self.num_nodes = config.num_nodes
        self.inputs, self.targets, self.seq_lenghth = batch_generator(data, self.batch_size)
        self.batch_num = len(self.inputs)
        self.cur_batch = 0

    def next_batch(self):
        x = self.inputs[self.cur_batch]
        y = self.targets[self.cur_batch]
        sl = self.seq_lenghth[self.cur_batch]
        self.cur_batch = (self.cur_batch +1) % self.batch_num
        batch_size = x.shape[0]
        num_steps = x.shape[1]
        return x, y, sl, batch_size, num_steps

def rank_eval(pred, true_labels, sl):
    mrr = 0
    ac1 = 0
    ac5 = 0
    ac10 = 0
    ac50 = 0
    ac100 = 0
    num_nodes = pred.shape[2]
    for i in range(len(sl)):
        length = sl[i]
        for j in range(length):
            y_pos = true_labels[i][j]
            predY = pred[i][j][y_pos]
            rank = 1.
            for k in range(num_nodes):
                if pred[i][j][k]> predY:
                    rank += 1.
            if rank <= 1:
                ac1 += 1./float(length)
            if rank <= 5:
                ac5 += 1./float(length)
            if rank <= 10:
                ac10 += 1./float(length)
            if rank <= 50:
                ac50 += 1./float(length)
            if rank <= 100:
                ac100 += 1./float(length)
            mrr += (1./rank)/float(length)
    return mrr, ac1, ac5, ac10, ac50, ac100

def args_setting(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lr", type=float, help="learning rate")
    parser.add_argument("-x", "--xdim", type=int, help="embedding dimension")
    parser.add_argument("-e", "--hdim", type=int, help="hidden dimension")
    parser.add_argument("-d", "--data", help="data name")
    parser.add_argument("-g", "--gpu", help="gpu id")
    parser.add_argument("-b", "--bs", type=int, help="batch size")
    parser.add_argument("-f", "--freq", type=int, help="validation frequency")
    parser.add_argument("-n", "--nepoch", type=int, help="number of training epochs")
    args = parser.parse_args()
    if args.lr:
        config.learning_rate = args.lr
    if args.xdim:
        config.embedding_dim = args.xdim
    if args.hdim:
        config.hidden_dim = args.hdim
    if args.bs:
        config.batch_size = args.bs
    if args.data:
        config.data_name = args.data
    if args.gpu:
        config.gpu_no = args.gpu
    if args.freq:
        config.valid_freq = args.freq
        config.patience = int(10/config.valid_freq) + 1
    if args.nepoch:
        config.num_epochs = args.nepoch
    return config

def train(argv):
    config = Config()
    config = args_setting(config)

    data_name = config.data_name
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_no
    num_epochs = config.num_epochs

    # data load
    train_data, valid_data, test_data, nodes, node_to_id = \
        read_raw_data(data_name)
    config.num_nodes = len(nodes)
    train_size = train_data[2]
    valid_size = valid_data[2]
    test_size = test_data[2]
    print (train_size, valid_size, test_size)
    #A = read_graph(data_name + '-graph', node_to_id)
    A = read_graph(data_name + '-minmax.csv', node_to_id)
    print(A.shape)
    input_train = Input(config, train_data)
    input_valid = Input(config, valid_data)
    input_test = Input(config, test_data)

    # Model create
    model = SNIDSA(config, A)
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=tfconfig)
    tf.set_random_seed(config.random_seed)

    # Parameter Initialization
    sess.run(tf.global_variables_initializer())

    # Record test results at best validation epoch with early stopping
    max_logits = float('inf')
    stop_count = 0
    best_mrr = 0
    best_ac1 = 0
    best_ac5 = 0
    best_ac10 = 0
    best_ac50 = 0
    best_ac100 = 0
    best_valid_epoch = 0

    # Print Training information
    train_info = "Data: {0}, Model: {1}, GPU Num: {2}, Learning Rate: {3:.3f}, Embedding Size: {4}, Hidden Size: {5}, Batch Size: {6}"
    print(train_info.format(config.data_name, config.model, config.gpu_no, config.learning_rate, config.embedding_dim, config.hidden_dim, config.batch_size))
    print('Start training...')

    ff = open('res.txt', 'w')

    # Training Process
    for epoch in range(num_epochs):
        epoch_logits = 0
        valid_logits = 0
        # test_logits = 0
        valid_mrr = 0
        valid_ac1 = 0
        valid_ac5 = 0
        valid_ac10 = 0
        valid_ac50 = 0
        valid_ac100 = 0
        test_mrr = 0
        test_ac1 = 0
        test_ac5 = 0
        test_ac10 = 0
        test_ac50 = 0
        test_ac100 = 0

        msg = 'Epoch ' + str(epoch+1) + '/' + str(num_epochs) + ' (Train)'
        for i in tqdm(range(input_train.batch_num), desc=msg):
            x_batch, y_batch, seq_length, batch_size, num_steps = input_train.next_batch()
            feed_dict = {model._inputs: x_batch, model._targets: y_batch, model._seqlen: seq_length, model.batch_size: batch_size, model.num_steps: num_steps}
            _, batch_cost = sess.run([model.optim, model.nll], feed_dict=feed_dict)
            epoch_logits += np.sum(batch_cost)
        msg = "Train NLL: {0:>6.3f}"
        print(msg.format(epoch_logits/float(train_size)))

        if (epoch+1)%config.valid_freq == 0:
            msg = 'Epoch ' + str(epoch+1) + '/' + str(num_epochs) + ' (Val.)'
            for j in tqdm(range(input_valid.batch_num), desc=msg):
                x_batch, y_batch, seq_length, batch_size, num_steps = input_valid.next_batch()
                feed_dict = {model._inputs: x_batch, model._targets: y_batch, model._seqlen: seq_length, model.batch_size: batch_size, model.num_steps: num_steps}
                valid_nll, valid_pred = sess.run([model.nll, model.pred], feed_dict=feed_dict)
                valid_logits += np.sum(valid_nll)
                mrr, ac1, ac5, ac10, ac50, ac100 = rank_eval(valid_pred, y_batch, seq_length)
                valid_mrr += mrr
                valid_ac1 += ac1
                valid_ac5 += ac5
                valid_ac10 += ac10
                valid_ac50 += ac50
                valid_ac100 += ac100

            msg = 'Epoch ' + str(epoch+1) + '/' + str(num_epochs) + ' (Test)'
            for k in tqdm(range(input_test.batch_num), desc=msg):
                x_batch, y_batch, seq_length, batch_size, num_steps = input_test.next_batch()
                feed_dict = {model._inputs: x_batch, model._targets: y_batch, model._seqlen: seq_length, model.batch_size: batch_size, model.num_steps: num_steps}
                test_pred = sess.run(model.pred, feed_dict=feed_dict)
                mrr, ac1, ac5, ac10, ac50, ac100 = rank_eval(test_pred, y_batch, seq_length)
                test_mrr += mrr
                test_ac1 += ac1
                test_ac5 += ac5
                test_ac10 += ac10
                test_ac50 += ac50
                test_ac100 += ac100

            msg = "Val. NLL: {0:>6.3f}"
            print(msg.format(valid_logits/float(valid_size)))

            msg = "Val. MRR: {0:>6.5f}, ACC1: {1:>6.5f}, ACC5: {2:>6.5f}, ACC10: {3:>6.5f}, ACC50: {4:>6.5f}, ACC100: {5:>6.5f}"
            print(msg.format( valid_mrr/float(valid_size), valid_ac1/float(valid_size), valid_ac5/float(valid_size), valid_ac10/float(valid_size), valid_ac50/float(valid_size), valid_ac100/float(valid_size)))
            ff.write(str(msg.format( valid_mrr/float(valid_size), valid_ac1/float(valid_size), valid_ac5/float(valid_size), valid_ac10/float(valid_size), valid_ac50/float(valid_size), valid_ac100/float(valid_size))) + '\n')

            msg = "Test MRR: {0:>6.5f}, ACC1: {1:>6.5f}, ACC5: {2:>6.5f}, ACC10: {3:>6.5f}, ACC50: {4:>6.5f}, ACC100: {5:>6.5f}"
            print(msg.format( test_mrr/float(test_size), test_ac1/float(test_size), test_ac5/float(test_size), test_ac10/float(test_size), test_ac50/float(test_size), test_ac100/float(test_size)))
            ff.write(str(msg.format( test_mrr/float(test_size), test_ac1/float(test_size), test_ac5/float(test_size), test_ac10/float(test_size), test_ac50/float(test_size), test_ac100/float(test_size))) + '\n\n')

            if valid_logits < max_logits: #Early stop with checking negative log-likelihood on validation set
                max_logits = valid_logits
                best_valid_epoch = epoch+1
                # Record test results at best validation epoch
                best_mrr = test_mrr
                best_ac1 = test_ac1
                best_ac5 = test_ac5
                best_ac10 = test_ac10
                best_ac50 = test_ac50
                best_ac100 = test_ac100
                stop_count = 0
                # To do: save_model()
            else:
                stop_count += 1

            if stop_count==config.patience:
                break

    print('Finish training...')

    print('Best valid negative log-likelihood at Epoch: %d ' % best_valid_epoch)

    msg = "Test MRR: {0:>6.5f}, ACC1: {1:>6.5f}, ACC5: {2:>6.5f}, ACC10: {3:>6.5f}, ACC50: {4:>6.5f}, ACC100: {5:>6.5f}"
    print(msg.format(best_mrr/float(test_size), best_ac1/float(test_size), best_ac5/float(test_size), best_ac10/float(test_size), best_ac50/float(test_size), best_ac100/float(test_size) ))

    # Save results of best validation model
    with open('results.txt', 'a') as f:
        f.write('Test results on ' + data_name + ':\n')
        f.write('MRR: '+ str(best_mrr/float(test_size)) + '\n')
        f.write('ACC1: '+ str(best_ac1/float(test_size)) + '\n')
        f.write('ACC5: '+ str(best_ac5/float(test_size)) + '\n')
        f.write('ACC10: '+ str(best_ac10/float(test_size)) + '\n')
        f.write('ACC50: '+ str(best_ac50/float(test_size)) + '\n')
        f.write('ACC100: '+ str(best_ac100/float(test_size)) + '\n\n')

    sess.close()

if __name__ == '__main__':
    train(sys.argv)