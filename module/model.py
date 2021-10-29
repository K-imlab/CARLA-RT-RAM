import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Activation, LSTM, Reshape, Conv2D, MaxPool2D, RepeatVector, GRU
import tensorflow.keras.backend as K
import os

import config

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def Net(man, purpose, args):
    ## input = hist
    hist_input = tf.keras.Input(shape=(args['in_length'], args['num_input']), name='hist_Input')
    hist = Dense(args['input_embedding_size'], name='hist_Embedding')(hist_input)
    hist = Activation(tf.nn.leaky_relu)(hist)

    hist, state_h, state_c = LSTM(args['encoder_size'], return_sequences=False, return_state=True,
                                  name='hist_LSTM_encode')(hist)
    encoder_state = [state_h, state_c]
    hist = Activation(tf.nn.leaky_relu)(hist)

    hist = Dense(args['dyn_embedding_size'], name='hist_dyn_Enbedding')(hist)
    hist = Activation(tf.nn.leaky_relu)(hist)

    ## input = nbr
    nbr_input = tf.keras.Input(shape=(args['grid_x'] * args['grid_y'], args['in_length'], args['num_input']),
                               name='nbr_Input')
    nbrs = Dense(args['input_embedding_size'], name='nbr_Embedding')(nbr_input)
    nbrs = Activation(tf.nn.leaky_relu)(nbrs)

    nbrs = Reshape((args['grid_x'] * args['grid_y'], args['in_length'] * args['input_embedding_size']),
                   name='nbr_Reshape')(nbrs)
    nbrs = Dense(args['encoder_size'], name='nbr_dyn_Enbedding')(nbrs)

    nbrs = Activation(tf.nn.leaky_relu)(nbrs)

    nbrs = Reshape((args['grid_y'], args['grid_x'], args['encoder_size']), name='nbr_reshape_conv_input')(nbrs)

    nbrs = Conv2D(args['soc_conv_depth'], args['soc_conv_filter'], name='Conv2D_1')(nbrs)
    nbrs = Activation(tf.nn.leaky_relu)(nbrs)
    nbrs = Conv2D(args['conv_3x1_depth'], args['conv_3x1_filter'], name='Conv2D_2')(nbrs)
    nbrs = Activation(tf.nn.leaky_relu)(nbrs)
    nbrs = MaxPool2D(args['maxpool_size'], padding='same')(nbrs)
    nbrs = Flatten()(nbrs)

    ## concatenate
    decode = tf.concat([hist, nbrs], axis=1)

    if man:
        maneuver = Dense(args['num_maneuver'], name='dense_maneuver')(decode)
        maneuver = Activation(tf.nn.softmax, name='output_man')(maneuver)

        decode = tf.concat([decode, maneuver], axis=1)

    decode = RepeatVector(n=args['out_length'])(decode)
    decode = LSTM(args['decoding_size'], return_sequences=True, name='LSTM_decode')(decode, initial_state=encoder_state)
    decode = Activation(tf.nn.leaky_relu)(decode)
    decode = Dense(args['num_output'], name='Output')(decode)
    decode = Activation(tf.nn.leaky_relu, name='output_traj')(decode)

    if man:
        if purpose == 'train':
            model = Model(inputs=[hist_input, nbr_input], outputs=[decode, maneuver], name='Model')
        elif purpose == 'test':
            model = Model(inputs=[hist_input, nbr_input], outputs=decode, name='Model')
        else:
            print('Please typing only "train" or "test" for purpose')

    else:
        model = Model(inputs=[hist_input, nbr_input], outputs=decode, name='Model')

    print(model.summary())

    return model


def loss(pred, true, x_weights=1):
    x_diff = (pred[:, :, 0] - true[:, :, 0]) * x_weights
    x_diff_sq = K.square(x_diff)
    y_diff = pred[:, :, 1] - true[:, :, 1]
    y_diff_sq = K.square(y_diff)
    sq = x_diff_sq + y_diff_sq
    sq_mean = K.mean(sq, axis=1)
    rmse = K.sqrt(sq_mean)
    rmse_mean = K.mean(rmse)
    return rmse_mean


def weight_load():
    args_train = {}
    args_train['optimizer'] = tf.keras.optimizers.Adam
    args_train['learning_rate'] = 0.001
    args_train['batch_size'] = 128
    args_train['epoch'] = 30

    model = tf.keras.models.load_model(
        os.path.join(config.ProjectRoot, "module", "vissim_model.h5"),
        custom_objects={"leaky_relu": tf.nn.leaky_relu, "loss": loss})
    model.compile(optimizer=args_train["optimizer"](learning_rate=args_train["learning_rate"]), loss=loss)
    return model

