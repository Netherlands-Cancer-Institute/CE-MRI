import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Maximum, Add, Input, Multiply, Subtract, Conv2D, Conv2DTranspose, UpSampling2D, concatenate, MaxPooling2D, Lambda, Reshape, LeakyReLU, BatchNormalization, Dense, Dropout, Activation
from attention_module import parameter_attention


def MixedFusion_block_d(x1, x2, x3, x4, x5, xx):
    init = RandomNormal(stddev=0.02)
    adc0_150 = (x2-x3)/(150-0)
    adc150_800 = (x3-x4)/(800-150)
    adc800_1500 = (x4-x5)/(1500-800)
    channel_n = x1.shape[-1]
    out_fusion_c = concatenate([x1, x2, x3, x4, x5, adc0_150,adc150_800,adc800_1500], axis=-1)
    out_fusion = parameter_attention(out_fusion_c)
    l1 = Conv2D(channel_n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(out_fusion)
    l2 = BatchNormalization()(l1)
    l3 = LeakyReLU(alpha=0.2)(l2)
    c = concatenate([l3, xx], axis=-1)
    d1 = Conv2D(channel_n * 2, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(c)
    d2 = BatchNormalization()(d1)
    d3 = LeakyReLU(alpha=0.2)(d2)
    out666 = d3
    return out666


def MixedFusion_block_u(x1, x2, x3, x4, x5, xx, skip):
    init = RandomNormal(stddev=0.02)
    adc0_150 = (x2-x3)/(150-0)
    adc150_800 = (x3-x4)/(800-150)
    adc800_1500 = (x4-x5)/(1500-800)
    channel_n = x1.shape[-1]
    out_fusion_c = concatenate([x1, x2, x3, x4, x5,adc0_150,adc150_800,adc800_1500], axis=-1)
    out_fusion = parameter_attention(out_fusion_c)
    l1 = Conv2D(channel_n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(out_fusion)
    l2 = BatchNormalization()(l1)
    l3 = Activation('relu')(l2)
    c = concatenate([l3, xx, skip], axis=-1)
    u1 = Conv2D(channel_n / 2, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(c)
    u2 = BatchNormalization()(u1)
    u3 = Activation('relu')(u2)
    out666 = u3
    return out666


def MixedFusion_block_0(x1, x2, x3, x4, x5):
    init = RandomNormal(stddev=0.02)
    adc0_150 = (x2-x3)/(150-0)
    adc150_800 = (x3-x4)/(800-150)
    adc800_1500 = (x4-x5)/(1500-800)
    channel_n = x1.shape[-1]
    out_fusion_c = concatenate([x1, x2, x3, x4, x5, adc0_150,adc150_800,adc800_1500], axis=-1)
    out_fusion = parameter_attention(out_fusion_c)
    l1 = Conv2D(channel_n, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(out_fusion)
    l2 = BatchNormalization()(l1)
    l3 = LeakyReLU(alpha=0.2)(l2)
    d1 = Conv2D(channel_n * 2, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(l3)
    d2 = BatchNormalization()(d1)
    d3 = LeakyReLU(alpha=0.2)(d2)
    out666 = d3
    return out666
