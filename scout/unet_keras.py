import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate
from tensorflow.keras import Model


def get_unet(path):
    base_filters = 16
    pool_size = (2, 2)

    conv_args = {'activation': 'relu',
                 'padding': 'same',
                 'kernel_regularizer': tf.keras.regularizers.L1L2(l2=1e-5)}

    inputs = Input((None, None, 1))

    # Encoder
    conv1 = Conv2D(base_filters, 3, **conv_args)(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = Conv2D(base_filters, 3, **conv_args)(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

    conv2 = Conv2D(base_filters * 2, 3, **conv_args)(pool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = Conv2D(base_filters * 2, 3, **conv_args)(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

    conv3 = Conv2D(base_filters * 4, 3, **conv_args)(pool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = Conv2D(base_filters * 4, 3, **conv_args)(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

    conv4 = Conv2D(base_filters * 8, 3, **conv_args)(pool3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = Conv2D(base_filters * 8, 3, **conv_args)(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=pool_size)(drop4)

    conv5 = Conv2D(base_filters * 16, 3, **conv_args)(pool4)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = Conv2D(base_filters * 16, 3, **conv_args)(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=pool_size)(drop5)

    conv6 = Conv2D(base_filters * 32, 3, **conv_args)(pool5)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = Conv2D(base_filters * 32, 3, **conv_args)(conv6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    drop6 = Dropout(0.5)(conv6)
    pool6 = MaxPooling2D(pool_size=pool_size)(drop6)

    conv7 = Conv2D(base_filters * 64, 3, **conv_args)(pool6)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = Conv2D(base_filters * 64, 3, **conv_args)(conv7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    drop7 = Dropout(0.5)(conv7)

    # Decoder
    up8 = tf.keras.layers.Conv2DTranspose(base_filters * 32, 2, strides=pool_size, **conv_args)(drop7)
    merge8 = concatenate([drop6,up8], axis = 3)
    conv8 = Conv2D(base_filters * 32, 3, **conv_args)(merge8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = Conv2D(base_filters * 32, 3, **conv_args)(conv8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)

    up9 = tf.keras.layers.Conv2DTranspose(base_filters * 16, 2, strides=pool_size, **conv_args)(conv8)
    merge9 = concatenate([drop5,up9], axis = 3)
    conv9 = Conv2D(base_filters * 16, 3, **conv_args)(merge9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = Conv2D(base_filters * 16, 3, **conv_args)(conv9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)

    up10 = tf.keras.layers.Conv2DTranspose(base_filters * 8, 2, strides=pool_size, **conv_args)(conv9)
    merge10 = concatenate([drop4,up10], axis = 3)
    conv10 = Conv2D(base_filters * 8, 3, **conv_args)(merge10)
    conv10 = tf.keras.layers.BatchNormalization()(conv10)
    conv10 = Conv2D(base_filters * 8, 3, **conv_args)(conv10)
    conv10 = tf.keras.layers.BatchNormalization()(conv10)

    up11 = tf.keras.layers.Conv2DTranspose(base_filters * 4, 2, strides=pool_size, **conv_args)(conv10)
    merge11 = concatenate([conv3,up11], axis = 3)
    conv11 = Conv2D(base_filters * 4, 3, **conv_args)(merge11)
    conv11 = tf.keras.layers.BatchNormalization()(conv11)
    conv11 = Conv2D(base_filters * 4, 3, **conv_args)(conv11)
    conv11 = tf.keras.layers.BatchNormalization()(conv11)

    up12 = tf.keras.layers.Conv2DTranspose(base_filters * 2, 2, strides=pool_size, **conv_args)(conv11)
    merge12 = concatenate([conv2,up12], axis = 3)
    conv12 = Conv2D(base_filters * 2, 3, **conv_args)(merge12)
    conv12 = tf.keras.layers.BatchNormalization()(conv12)
    conv12 = Conv2D(base_filters * 2, 3, **conv_args)(conv12)
    conv12 = tf.keras.layers.BatchNormalization()(conv12)

    up13 = tf.keras.layers.Conv2DTranspose(base_filters, 2, strides=pool_size, **conv_args)(conv12)
    merge13 = concatenate([conv1,up13], axis = 3)
    conv13 = Conv2D(base_filters, 3, **conv_args)(merge13)
    conv13 = tf.keras.layers.BatchNormalization()(conv13)
    conv13 = Conv2D(base_filters, 3, **conv_args)(conv13)
    conv13 = tf.keras.layers.BatchNormalization()(conv13)

    conv14 = Conv2D(1, 1, activation='sigmoid')(conv13)

    unet = Model(inputs, conv14)
    unet.load_weights(path)

    return unet