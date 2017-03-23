import tensorflow as tf
from collections import defaultdict
from util import *
#from tensorflow.python.ops import seq2seq as s2s
from tensorflow.contrib import layers
import numpy as np
import random
from tensorflow.python.client import device_lib

PADDING_SAME = 'SAME'
PADDING_VALID = 'VALID'
tf_sigmoid = tf.nn.sigmoid
tf_tanh = tf.nn.tanh
tf_elu = tf.nn.elu
tf_relu = tf.nn.relu


def tf_leaky_relu(leak=0.2):
    return lambda x: tf.maximum(x, leak*x)


def tf_dense(input_, output_size, stddev=0.02, bias_start=0.0, reuse=False, name=None,
             activation=None, bn=False):
    shape = input_.get_shape().as_list()
    scope = name
    with tf.variable_scope(scope or "Dense", reuse=reuse):
        matrix = tf.get_variable("matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start, dtype=tf.float32))
        result = tf.matmul(input_, matrix) + bias
    if bn:
        result = tf_batch_norm(result, mode='1d', name=name+"_bn", reuse=reuse)
    if activation is not None:
        result = activation(result)
    return result
dense = tf_dense


def tf_seq2seq(encoder_inputs, decoder_inputs, loop_function=None, cell=None, name="seq2seq",
               use_loop_function=False):
    """
    :param encoder_inputs: A list of 2D Tensors [batch_size x input_size]
    :param decoder_inputs: A list of 2D Tensors [batch_size x input_size]
    :param loop_function:
    :param cell:
    :param use_loop_function:
    :param scope:
    :return:
    """
    scope = name
    if loop_function is None or use_loop_function is False:
        return s2s.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, scope=scope)
    else:
        return s2s.tied_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                                    loop_function=loop_function, scope=scope)


def tf_input(shape, name, dtype=tf.float32, scope=None):
    if scope is None:
        return tf.placeholder(dtype, shape=shape, name=name)
    else:
        with tf.variable_scope(scope):
            return tf.placeholder(dtype, shape=shape, name=name)


def tf_var(shape, name, dtype=tf.float32, scope=None,
           reuse=False,
           regularizer=None,
           initializer=tf.random_normal_initializer(0.001)):
    if scope is None:
        return tf.get_variable(name=name, shape=shape, dtype=dtype,
                               initializer=initializer, regularizer=regularizer)
    else:
        with tf.variable_scope(scope, reuse=reuse):
            return tf.get_variable(name=name, shape=shape, dtype=dtype,
                                   initializer=initializer, regularizer=regularizer)


def tf_basic_vae(input_dim=100, encoder=None, decoder=None, z_dim=100,
                 optimizer=tf.train.AdamOptimizer(0.005),
                 reuse=False, name="tf_basic_vae"):
    """
    TODO
    :return:
    """
    scope = name

    def n(message):
        return "{0}_{1}".format(scope, message)

    if isinstance(input_dim, int):
        input_dim = (input_dim, )

    if isinstance(input_dim, list):
        input_dim = tuple(input_dim)

    with tf.variable_scope(scope, reuse=reuse):
        input_var = tf_input([None] + to_list(input_dim), n('input_var'))
        batch_size = input_var.get_shape()[0]
        epsilon = tf_input((batch_size, z_dim), n('epsilon'))
        with tf.variable_scope("encoder", reuse=False):
            mu, sigma = encoder(input_var)
        z = mu + tf.mul(tf.exp(0.5 * sigma), epsilon)
        with tf.variable_scope("decoder", reuse=False):
            recon_input_var, recon_input_var_activated = decoder(z)
        with tf.variable_scope("decoder", reuse=True):
            gen_input_var, gen_input_var_activated = decoder(epsilon)
        kld = -0.5 * tf.reduce_sum(1 + sigma - tf.pow(mu, 2) - tf.exp(sigma), reduction_indices=1)
        bce = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                            tf_flatten_for_dense(recon_input_var), tf_flatten_for_dense(input_var)),
                            reduction_indices=1)
        loss = tf.reduce_mean(kld + bce)
        _ = tf.scalar_summary(n("loss"), loss)
        summary_op = tf.merge_all_summaries()
        train_op = optimizer.minimize(loss)
        return {
            'op': [train_op, summary_op],
            'loss': loss,
            'output': [recon_input_var, recon_input_var_activated,
                       gen_input_var, gen_input_var_activated],
            'input': [input_var, epsilon],
            'kld': kld,
            'bce': bce,
            'mu': mu,
            'sigma': sigma,
            'epsilon': epsilon,
            'input_var': input_var,
            'generation': gen_input_var_activated,
            'train': train_op
        }


class VaeModel:
    def __init__(self, **kwargs):
        m = tf_basic_vae(**kwargs)
        self.train_op = m['op'][0]
        self.generation_output = m['generation']
        self.input = m['input_var']
        self.mu = m['mu']
        self.sigma = m['sigma']
        self.epsilon = m['epsilon']
        self.loss = m['loss']
        self.summary_op = m['op'][1]


def tf_load_model(session, saver, model_name, log=print, save_dir=model_save_temp_folder__):
    model_dir = join(save_dir, model_name)
    mkdir(model_dir)
    model_path = join(model_dir, 'model.ckpt.meta')
    ckpt_path = model_path
    if isfile(ckpt_path):
        log("Restoring saved parameters")
        saver.restore(session, join(model_dir, 'model.ckpt'))
    else:
        log("Initializing parameters")
        tf_init(session)


def tf_save_model(session, saver, model_name, save_dir=model_save_temp_folder__, log=print):
    log("Saving model {0}".format(model_name))
    model_dir = join(save_dir, model_name)
    mkdir(model_dir)
    model_path = join(model_dir, 'model.ckpt')
    saver.save(session, model_path)
    log("finish!")


class ModelSaver:
    def __init__(self, session, model_name, log=print, save_dir=model_save_temp_folder__):
        self.saver = tf.train.Saver()
        self.model_name = model_name
        self.session = session
        self.save_dir = save_dir
        self.log = log

    def load(self):
        tf_load_model(self.session, self.saver, self.model_name, log=self.log, save_dir=self.save_dir)

    def save(self):
        tf_save_model(self.session, self.saver, self.model_name, log=self.log, save_dir=self.save_dir)


def tf_need_test(step, freq):
    r = step % freq
    return r == 0


def tf_time():
    return datetime.now()


def tf_log(i, loss, n_iter=1000, start_time=None, log=print, message=""):
    now = tf_time()
    run_time = now - start_time if start_time is not None else None
    log(message + "{0}/{1} iterations, loss = {2}, cost time = {3}".format(i, n_iter, loss, run_time))


def tf_dense_with_gated_activation(input_var, output_size=None, name="tf_dense_with_gated_activation",
                                   reuse=False, stddev=1,
                                   bias_start=0.0, after_activation=None, residual=False):
    scope = name
    if residual:
        output_size = input_var.get_shape()[1]
    else:
        if output_size is None:
            raise Exception("Output size has to be specified if it is not a residual layer")
    with tf.variable_scope(scope, reuse=reuse):
        branch_sigmoid = tf.nn.sigmoid(tf_dense(input_var, output_size,
                                                name="branch_sigmoid", reuse=reuse, stddev=stddev,
                                                bias_start=bias_start))
        branch_tanh = tf.nn.tanh(tf_dense(input_var, output_size,
                                          name="branch_tanh", reuse=reuse, stddev=stddev, bias_start=bias_start))
        result = tf.mul(branch_tanh, branch_sigmoid, name="gate")
        if residual:
            result = tf_dense(result, output_size, name="residual_connection",
                              reuse=reuse, stddev=stddev, bias_start=bias_start)
            result = tf.add(result, input_var)
            if after_activation is not None:
                result = after_activation(result)
        else:
            if after_activation is not None:
                raise Exception("after_activation only works with residual blocks!")
        return result


def tf_deconv2d(input_, output_shape,
                k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, bn=False,
                name="deconv2d", padding='SAME', reuse=False, activation=None):
    """
    shape calculation in https://gist.github.com/BenBBear/19df8ac8ff17926f2659962f8b1dc5b1
    """
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        batch_size = tf_get_batch_size(input_)
        output_shape = to_list(output_shape)
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[batch_size, ] + output_shape,
                                        strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        deconv = tf.reshape(deconv, [batch_size, ] + output_shape)
    if bn:
        deconv = tf_batch_norm(deconv, mode='2d', name=name+"_bn", reuse=reuse)
    if activation is not None:
        deconv = activation(deconv)
    return deconv


def tf_conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
              name="conv2d", reuse=False, activation=None, padding='SAME', bn=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
    if bn:
        conv = tf_batch_norm(conv, mode='2d', name=name+"_bn", reuse=reuse)
    if activation is not None:
        conv = activation(conv)
    return conv


def tf_get_batch_size(input_var):
    return tf.shape(input_var)[0]


def tf_flatten_for_dense(input_var):
    shape_list = list(input_var.get_shape()[1:])
    return tf.reshape(input_var, [-1, list_prod(shape_list)])


def tf_reshape_for_conv(input_var, shape):
    return tf.reshape(input_var, [-1, ] + list(shape))

tf_flatten = layers.flatten


def int_shape(x):
    '''Returns the shape of a tensor as a tuple of
    integers or None entries.
    Note that this function only works with TensorFlow.
    '''
    shape = x.get_shape()
    return tuple([i.__int__() for i in shape])


def tf_up_sampling_2d(input_var, height_factor, width_factor):
    original_shape = int_shape(input_var)
    new_shape = tf.shape(input_var)[1:3]
    new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
    x = tf.image.resize_nearest_neighbor(input_var, new_shape)
    x.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                 original_shape[2] * width_factor if original_shape[2] is not None else None, None))
    return x


def tf_repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    '''
    x_shape = x.get_shape().as_list()
    # slices along the repeat axis
    splits = tf.split(axis, x_shape[axis], x)
    # repeat each slice the given number of reps
    x_rep = [s for s in splits for _ in range(rep)]
    return tf.concat(axis, x_rep)


def tf_up_sampling_3d(input_var, depth_factor, height_factor, width_factor):
    output = tf_repeat_elements(input_var, depth_factor, axis=1)
    output = tf_repeat_elements(output, height_factor, axis=2)
    output = tf_repeat_elements(output, width_factor, axis=3)
    return output


def tf_conv3d(input_, output_dim, k_d=5, k_h=5, k_w=5, d_d=2, d_h=2, d_w=2, stddev=0.02,
              name="conv3d", reuse=False, activation=None, padding='SAME', bn=False):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
    if bn:
        conv = tf_batch_norm(conv, mode='3d', name=name+"_bn", reuse=reuse)
    if activation is not None:
        conv = activation(conv)
    return conv


def tf_deconv3d(input_, output_shape,
                k_d=5, k_h=5, k_w=5, d_d=2, d_h=2, d_w=2, stddev=0.02, bn=False,
                name="deconv3d", padding='SAME', reuse=False, activation=None):
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        batch_size = tf_get_batch_size(input_)
        output_shape = to_list(output_shape)
        w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=[batch_size, ] + output_shape,
                                        strides=[1, d_d, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        deconv = tf.reshape(deconv, [batch_size, ] + output_shape)
    if bn:
        deconv = tf_batch_norm(deconv, mode='3d', name=name+"_bn", reuse=reuse)
    if activation is not None:
        deconv = activation(deconv)
    return deconv


def tf_gan(input_dim=100, z_dim=100, d_net=None, g_net=None,
           optimizer=lambda: tf.train.AdamOptimizer(0.0002, beta1=0.5),
           feature_matching=None,
           feature_matching_weight=1.0,  # feature matching loss completely dominates the loss_gen
           conditional=None,
           conditional_dim=10,
           reuse=False, name="tf_basic_gan"):
    """
    second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      "scope/prefix/for/second/vars")
    """
    def n(message):
        return "{0}--{1}".format(message, name)
    input_shape = to_list(input_dim)
    input_data = tf_input([None, ] + input_shape, name=n("input_images"))

    # batch_size = tf_get_batch_size(input_data)
    # input_label = tf_input([None, 1], name=n("input_labels"))
    input_z = tf_input([None, z_dim],  name=n("uniform_z"))
    input_label = None
    if conditional:
        input_label = tf_input([None,] + to_list(conditional_dim), name=n("input_labels"))

    g_net_scope = n("g_net")
    d_net_scope = n("d_net")

    with tf.variable_scope(g_net_scope, reuse=reuse) as vs:
        output_gen = g_net(input_z, reuse=reuse, label=input_label)
        variables_gen = tf_get_variables_under_scope(vs)

    with tf.variable_scope(d_net_scope, reuse=reuse) as vs:
        if not feature_matching:
            discriminator_no_sigmoid, discriminator = d_net(input_data, reuse, label=input_label)
            feature_matching_input_data = None
        else:
            discriminator_no_sigmoid, discriminator, feature_matching_input_data = d_net(input_data,
                                                                                         reuse, label=input_label)
        variables_dis = tf_get_variables_under_scope(vs)

    with tf.variable_scope(d_net_scope, reuse=reuse) as vs:
        # set reuse to true will cause get_variable problem,
        # https://github.com/tensorflow/tensorflow/issues/5827
        if not feature_matching:
            gan_net_no_sigmoid, gan_net = d_net(output_gen, reuse=True, label=input_label)
            feature_matching_gan = None
        else:
            gan_net_no_sigmoid, gan_net, feature_matching_gan = d_net(output_gen, reuse=True, label=input_label)

    variables_all = variables_gen + variables_dis
    loss_dis_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_no_sigmoid,
                                                                           labels=tf.ones_like(discriminator)))
    loss_dis_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_net_no_sigmoid,
                                                                           labels=tf.zeros_like(gan_net)))
    loss_dis = loss_dis_real + loss_dis_fake
    loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_net_no_sigmoid,
                                                                      labels=tf.ones_like(gan_net_no_sigmoid)))
    if feature_matching:
        if feature_matching_weight > 1.0:
            feature_matching_weight = 1.0
        elif feature_matching_weight < 0:
            feature_matching_weight = 0.0
        gan_fm_output = feature_matching_gan
        input_data_fm_output = feature_matching_input_data
        loss_gen_feature_match = tf.reduce_mean(abs(tf.reduce_mean(gan_fm_output, 0)
                                                    - tf.reduce_mean(input_data_fm_output, 0)))
        loss_gen = (1-feature_matching_weight) * loss_gen + \
            feature_matching_weight * loss_gen_feature_match

    # print(variables_gen)
    # print(variables_dis)
    train_gen_op = optimizer().minimize(loss_gen, var_list=variables_gen)
    train_dis_op = optimizer().minimize(loss_dis, var_list=variables_dis)
    train_whole_op = optimizer().minimize(loss_gen, var_list=variables_all)
    gen_loss_sum = tf.summary.scalar(n("loss_gen"), loss_gen)
    d_dis_output_sum = tf.summary.histogram(n("D dis"), discriminator)
    d_gen_output_sum = tf.summary.histogram(n("D gen"), gan_net)
    dis_loss_sum = tf.summary.scalar(n("loss_dis"), loss_dis)
    dis_loss_real_sum = tf.summary.scalar(n("loss_dis_real"), loss_dis_real)
    dis_loss_fake_sum = tf.summary.scalar(n("loss_dis_fake"), loss_dis_fake)
    z_sum = tf.summary.histogram("z", input_z)
    summary_dis_op = tf.summary.merge([z_sum, d_dis_output_sum, dis_loss_real_sum, dis_loss_fake_sum, dis_loss_sum])
    summary_gen_op = tf.summary.merge([z_sum, d_gen_output_sum, gen_loss_sum, dis_loss_fake_sum])
    return {
        'op': [train_dis_op, train_gen_op, train_whole_op, summary_dis_op, summary_gen_op],
        'loss': [loss_dis, loss_gen, loss_dis_fake, loss_dis_real],
        'input': [input_data, input_z, input_label],
        'gan_net': [gan_net_no_sigmoid, gan_net],
        'generation': output_gen,
        'dis_net': [discriminator_no_sigmoid, discriminator],
        'variables': [variables_dis, variables_gen, variables_all],
        'scope':{
            'g': g_net_scope,
            'd': d_net_scope
        }
    }


class GanModel:
    G = 0
    D = 1
    DG = 2

    Message = ["Gan - Generator: ", "Gan - Discriminator: ", "Gan - Dis + Gen: "]

    def __init__(self, **kwargs):
        k = kwargs
        k = defaultdict(lambda: None, k)
        m = tf_gan(input_dim=k['input_dim'], z_dim=k['z_dim'], d_net=k['d_net'], g_net=k['g_net'],
                   optimizer=k['optimizer'], reuse=k['reuse'], name=k['name'],
                   feature_matching=k['feature_matching'], feature_matching_weight=k['feature_matching_weight'],
                   conditional=k['conditional'], conditional_dim=k['conditional_dim'])
        self.conditional = k['conditional']
        self.scope = m['scope']
        self.feature_matching = k['feature_matching']
        self.z_dim = kwargs['z_dim']
        self.train_dis_op, self.train_gen_op, self.train_whole_op, self.summary_dis_op, self.summary_gen_op = m['op']
        self.loss_dis, self.loss_gen, self.loss_dis_fake, self.loss_dis_real = m['loss']
        self.input_data, self.input_z, self.input_label = m['input']
        self.gan_net_no_sigmoid, self.gan_net = m['gan_net']
        self.discriminator_no_sigmoid, self.discriminator = m['dis_net']
        self.generation = m['generation']
        self.variables_dis, self.variables_gen, self.variables_all = m['variables']
        self.session = kwargs['session']
        self.next_batch = kwargs['next_batch']
        self.batch_size = kwargs['batch_size']
        self.input_shape = to_list(kwargs['input_dim'])
        self.log = kwargs['log']
        self.model_name = kwargs['name']
        self.test_fun = kwargs['test_fun']
        self.test_freq = kwargs['test_freq']
        self.save_freq = kwargs['save_freq']
        self.saver = ModelSaver(self.session, self.model_name, log=self.log)
        self.model_path = make_model_path(self.model_name)
        summary_path = join(self.model_path, 'summary')
        mkdir(summary_path)
        self.summary_writer = tf.summary.FileWriter(summary_path, self.session.graph)
        self.saver.load()

    def generate_data(self, batch_size=None, label=None):
        if batch_size is None:
            batch_size = self.batch_size
        feed_dict = {
            self.input_z: self.random_z(batch_size)
        }
        if self.conditional:
            feed_dict[self.input_label] = label
        return self.session.run(self.generation, feed_dict=feed_dict)

    def random_z(self, batch_size):
        return np.random.uniform(-1, 1, (batch_size, self.z_dim)).astype(np.float32)

    def generate_mix_sample(self):
        real_images, _ = self.next_batch(self.batch_size)
        real_images = real_images.reshape([real_images.shape[0], ] + self.input_shape)
        generated_images = self.generate_data()
        label = np.zeros([2 * self.batch_size, 1])
        label[:self.batch_size, 0] = 1  # real images
        return tf_concat_batch(real_images, generated_images), label

    def test(self, stamp='id'):
        gen_volumes = self.generate_data(batch_size=4)
        gen_volumes = numpy_array_spiking(gen_volumes)
        output_path = join(make_model_path(self.model_name), 'testing')
        mkdir(output_path)
        plot_grid_volume(gen_volumes, dim=(1, 4), volume_shape=self.input_shape[:-1],
                         save_to=join(output_path, 'testing_volume_{0}.png'.format(stamp)))
        savemat(join(output_path, '{0}_volume'.format(stamp)), {
            'instance': gen_volumes
        })
        return gen_volumes

    def train(self, steps=[('D', 100)] + 3*[('G', 1000, 0.5, 1), ('D', 400)], log=print, message="",
              save=False, test=True):
        d_loss_list = []
        d_loss_real_list = []
        d_loss_fake_list = []
        g_loss_list = []
        all_loss = [["d_loss", d_loss_list], ["g_loss", g_loss_list],
                    ["d_loss_real", d_loss_real_list], ["d_loss_fake", d_loss_fake_list]]
        global_n_iter = 0
        for step in steps:
            for i in tqdm(range(step[1]), desc=step[0]):
                global_n_iter += 1
                start_time = tf_time()
                batch_z = self.random_z(self.batch_size)
                batch_images, batch_labels = self.next_batch(self.batch_size)
                batch_images = batch_images.reshape([batch_images.shape[0], ] + self.input_shape)
                feed_dict = {
                        self.input_data: batch_images,
                        self.input_z: batch_z
                    }
                if self.conditional:
                    feed_dict[self.input_label] = batch_labels
                _, d_loss, d_loss_real, d_loss_fake, summary_str = self.session.run(
                    [self.train_dis_op, self.loss_dis, self.loss_dis_real, self.loss_dis_fake, self.summary_dis_op],
                    feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str)
                d_loss_list.append(d_loss)
                d_loss_fake_list.append(d_loss_fake)
                d_loss_real_list.append(d_loss_real)
                if step[0] == 'G':
                    train_op = self.train_whole_op
                    if random.random() < step[2]:
                        train_op = self.train_gen_op
                    g_feed_dict = {
                        self.input_z:  batch_z
                    }
                    if self.feature_matching:
                        g_feed_dict[self.input_data] = batch_images
                    if self.conditional:
                        g_feed_dict[self.input_label] = batch_labels
                    for _ in range(step[3]):
                        _, g_loss, summary_str = self.session.run([train_op, self.loss_gen, self.summary_gen_op],
                                                                  feed_dict=g_feed_dict)
                        self.summary_writer.add_summary(summary_str)
                        g_loss_list.append(g_loss)
                    if tf_need_test(i, self.test_freq):
                        if test:
                            self.test_fun(model=self, n_iter=global_n_iter, losses=all_loss)
                    if tf_need_test(i, self.save_freq):
                        if save:
                            self.saver.save()
                if log is not None:
                    n_step = step[1]
                    log("--- D-G --- ")
                    tf_log(i, d_loss_list[-1], n_iter=n_step, start_time=start_time, log=log,
                           message=message + "Discriminator: ")
                    tf_log(i, g_loss_list[-1], n_iter=n_step, start_time=start_time, log=log,
                           message=message + "GAN: ")

        if save:
            self.saver.save()
        if test:
            self.test_fun(model=self, n_iter=global_n_iter, losses=all_loss)
        return all_loss


def tf_concat_batch(*args):
    return np.concatenate(args)


def tf_max_pool2d(input_, d_h=3, d_w=3, name="pool2d",  padding='SAME'):
    window = [1, d_h, d_w, 1]
    return tf.nn.max_pool(input_, window, window, padding=padding, name=name)


def tf_max_pool3d(input_, d_d=3, d_h=3, d_w=3, name="pool3d", padding='SAME'):
    window = [1, d_d, d_h, d_w, 1]
    return tf.nn.max_pool3d(input_, window, window, padding=padding, name=name)


def tf_batch_norm(input_var, epsilon=1e-5, decay=0.9, mode='3d', name="batch_norm_2d", reuse=False):
    axes = [0, 1, 2]
    if mode == '3d':
        axes = [0, 1, 2, 3]
    elif mode == '2d':
        axes = [0, 1, 2]
    elif mode == '1d':
        axes = [0, 1]

    with tf.variable_scope(name, reuse=reuse):
        shape = input_var.get_shape().as_list()
        beta = tf.get_variable("beta", [shape[-1]],
                               initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
        batch_mean, batch_var = tf.nn.moments(input_var, axes, name='moments')
    with tf.variable_scope("ExponentialMovingAverage", reuse=None):
        ema = tf.train.ExponentialMovingAverage(decay=decay)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    with tf.control_dependencies([ema_apply_op]):  # apply_op must be computed before the mean and var get used.
        mean, var = tf.identity(batch_mean), tf.identity(batch_var)
    normed = tf.nn.batch_normalization(input_var, mean, var, beta, gamma, epsilon, name="batch_normalization")
    return normed


def tf_dropout(input_var, drop_prob=0.5, name="dropout"):
    return tf.nn.dropout(input_var, 1-drop_prob, name=name)


def tf_soften_categorical_labels(y, l=0.1, h=0.9):
    if l >= h:
        raise Exception("soften_categorical_labels: l must be smaller than h! ")
    tf_l = tf.constant(l)
    tf_h = tf.constant(h)
    y *= (h-l)
    y += l
    return y


def tf_get_variables_under_scope(vs):
    try:
        key = tf.GraphKeys.GLOBAL_VARIABLES
    except AttributeError:
        key = tf.GraphKeys.VARIABLES
    return tf.get_collection(key, scope=vs.name)


def tf_num_inputs(h):
    input_shape = h.get_shape()[1:]
    return list_prod(input_shape, dtype=int)


def tf_dimension_to_int(d):
    if d.value is None:
        return d
    else:
        return int(d)


def tf_mini_batch_layer(input_, num_mini_batch_features=100, distance_dim_per_feature=5,
                        name="mini_batch_layer", reuse=False):
    """
    Example:     h = tf_mini_batch_layer(h, num_mini_batch_features=h.get_shape()[1],
                            distance_dim_per_feature=25, reuse=reuse)
    写了这个，对tensor操作的理解都大大加深了。。。。！
        how to compute the mini batch feature
        result = []
        for b1=1:feature_size
           for m=1:batch_size
	            for n=1:batch_size
                    result[m, b] += exp(- ||data[m, b, :]  -  data[n, b, :]||)

        another way to write it
        result = []
        for b=1:feature_size
          for m=1:batch_size
              result[m, b] += sum(sum(exp(- | | data[m, b, :1] - data[:2, b, :1] | |)))
    """
    with tf.variable_scope(name, reuse=reuse):
        if len(input_.get_shape()) > 2:
            input_ = tf_flatten_for_dense(input_)
        # batch_size x batch_feature_size x dim_per_feature
        batch_size = tf_get_batch_size(input_)
        num_mini_batch_features = int(num_mini_batch_features)
        x = tf_dense(input_, num_mini_batch_features * distance_dim_per_feature, reuse=reuse)
        activation = tf.reshape(x, [batch_size, num_mini_batch_features, distance_dim_per_feature])
        abs_difference = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) -
                                              tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
        mask = 1.0 - tf.expand_dims(tf.eye(batch_size), 1)
        masked = tf.exp(-abs_difference) * mask

        f1 = tf.reduce_sum(masked, 2) / \
            tf.reduce_sum(mask)
        mini_batch_features = [f1, ]
        return tf.concat(1, [input_] + mini_batch_features)


def tf_decaying_lr(lr, num_step_per_decay=100, decay_rate=0.99, name="decaying_learning_rate"):
    """
    https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/models/image/cifar10/cifar10.py
    max_steps = 1000000 ==> batch_number
    num_example_per_epoch = 50000
    num_batch_per_epoch = 390
    max_num_epoch = 1000000/390 = 2564 epoches.
    因此是 350 epoch -> per decay
    """
    global_step = tf.get_variable(name+'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    # decay_at_step 通常等于 int(num_batches_per_epoch * num_epoch_per_decay)
    return tf.train.exponential_decay(lr,
                                      global_step,
                                      num_step_per_decay,
                                      decay_rate,
                                      staircase=True)


def tf_get_available_device_list(dtype='GPU'):
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == dtype]


def tf_init(session):
    session.run(tf.global_variables_initializer())


def conv2d_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[1], x_shapes[2], y_shapes[3]])])


def conv3d_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(4, [x, y*tf.ones([x_shapes[1], x_shapes[2], x_shapes[3], y_shapes[4]])])


