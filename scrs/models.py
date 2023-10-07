import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Maximum, Add, Input, Multiply, Subtract, Conv2D, Conv2DTranspose, UpSampling2D, concatenate, MaxPooling2D, Lambda, Reshape, LeakyReLU, BatchNormalization, Dense, Dropout, Activation
from numpy import load
from config import LOSSES_WEIGHTS
from attention_module import parameter_attention
from fusion_module import MixedFusion_block_d, MixedFusion_block_u, MixedFusion_block_0

def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2, X11, X12, X13, X14 = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5']
    X1 = X1 / 3000
    X2 = X2 / 3000
    X11 = X11 / 3000
    X12 = X12 / 3000
    X13 = X13 / 3000
    X14 = X14 / 3000
    return [X1, X11, X12, X13, X14, X2]
  

def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    #in_src_image1 = Input(shape=image_shape)
    #in_src_image2 = Input(shape=image_shape)
    #in_src_image3 = Input(shape=image_shape)
    #in_src_image4 = Input(shape=image_shape)
    #in_src_image5 = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    #merged = Concatenate()([in_src_image1, in_src_image2, in_src_image3, in_src_image4, in_src_image5, in_target_image]) #!!!
    #merged = in_target_image
    # C64
    d = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(in_target_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(1024, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (3, 3), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    #model = Model([in_src_image1, in_src_image2, in_src_image3, in_src_image4, in_src_image5, in_target_image], patch_out)
    model = Model(in_target_image, patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[1])
    return model

def define_reconstruction(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input, MODALITY 1
    in_src_image1 = Input(shape=image_shape)
    # C64
    d1_0 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(in_src_image1)
    d1_0 = BatchNormalization()(d1_0)
    d1_0 = LeakyReLU(alpha=0.2)(d1_0)
    d1_0 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d1_0)
    d1_0 = BatchNormalization()(d1_0)
    d2_0 = LeakyReLU(alpha=0.2)(d1_0)
    # C128
    d3_0 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d2_0)
    d3_0 = BatchNormalization()(d3_0)
    d3_0 = LeakyReLU(alpha=0.2)(d3_0)
    d3_0 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d3_0)
    d3_0 = BatchNormalization()(d3_0)
    d4_0 = LeakyReLU(alpha=0.2)(d3_0)
    # C256
    d5_0 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d4_0)
    d5_0 = BatchNormalization()(d5_0)
    d5_0 = LeakyReLU(alpha=0.2)(d5_0)
    d5_0 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d5_0)
    d5_0 = BatchNormalization()(d5_0)
    d6_0 = LeakyReLU(alpha=0.2)(d5_0)
    # C512
    d7_0 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d6_0)
    d7_0 = BatchNormalization()(d7_0)
    d7_0 = LeakyReLU(alpha=0.2)(d7_0)
    d7_0 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d7_0)
    d7_0 = BatchNormalization()(d7_0)
    d8_0 = LeakyReLU(alpha=0.2)(d7_0)

    # bottleneck, no batch norm and relu
    b1_0 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d8_0)
    b2_0 = Activation('relu')(b1_0)

    # upsampling
    # c512
    u1_0 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(b2_0)
    u2_0 = BatchNormalization()(u1_0)
    u3_0 = Activation('relu')(u2_0)
    # c256
    u4_0 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u3_0)
    u5_0 = BatchNormalization()(u4_0)
    u6_0 = Activation('relu')(u5_0)
    # c128
    u7_0 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u6_0)
    u8_0 = BatchNormalization()(u7_0)
    u9_0 = Activation('relu')(u8_0)
    # c64
    u10_0 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u9_0)
    u11_0 = BatchNormalization()(u10_0)
    u12_0 = Activation('relu')(u11_0)

    u13_0 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u12_0)
    out_image_0 = Activation('sigmoid')(u13_0)

    # source image input, MODALITY 2
    in_src_image2 = Input(shape=image_shape)
    # C64
    d1_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(in_src_image2)
    d1_1 = BatchNormalization()(d1_1)
    d1_1 = LeakyReLU(alpha=0.2)(d1_1)
    d1_1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d1_1)
    d1_1 = BatchNormalization()(d1_1)
    d2_1 = LeakyReLU(alpha=0.2)(d1_1)
    # C128
    d3_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d2_1)
    d3_1 = BatchNormalization()(d3_1)
    d3_1 = LeakyReLU(alpha=0.2)(d3_1)
    d3_1 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d3_1)
    d3_1 = BatchNormalization()(d3_1)
    d4_1 = LeakyReLU(alpha=0.2)(d3_1)
    # C256
    d5_1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d4_1)
    d5_1 = BatchNormalization()(d5_1)
    d5_1 = LeakyReLU(alpha=0.2)(d5_1)
    d5_1 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d5_1)
    d5_1 = BatchNormalization()(d5_1)
    d6_1 = LeakyReLU(alpha=0.2)(d5_1)
    # C512
    d7_1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d6_1)
    d7_1 = BatchNormalization()(d7_1)
    d7_1 = LeakyReLU(alpha=0.2)(d7_1)
    d7_1 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d7_1)
    d7_1 = BatchNormalization()(d7_1)
    d8_1 = LeakyReLU(alpha=0.2)(d7_1)

    # bottleneck, no batch norm and relu
    b1_1 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d8_1)
    b2_1 = Activation('relu')(b1_1)

    # upsampling
    # c512
    u1_1 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(b2_1)
    u2_1 = BatchNormalization()(u1_1)
    u3_1 = Activation('relu')(u2_1)
    # c256
    u4_1 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u3_1)
    u5_1 = BatchNormalization()(u4_1)
    u6_1 = Activation('relu')(u5_1)
    # c128
    u7_1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u6_1)
    u8_1 = BatchNormalization()(u7_1)
    u9_1 = Activation('relu')(u8_1)
    # c64
    u10_1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u9_1)
    u11_1 = BatchNormalization()(u10_1)
    u12_1 = Activation('relu')(u11_1)

    u13_1 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u12_1)
    out_image_1 = Activation('sigmoid')(u13_1)

    # source image input, MODALITY 3
    in_src_image3 = Input(shape=image_shape)
    # C64
    d1_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(in_src_image3)
    d1_2 = BatchNormalization()(d1_2)
    d1_2 = LeakyReLU(alpha=0.2)(d1_2)
    d1_2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d1_2)
    d1_2 = BatchNormalization()(d1_2)
    d2_2 = LeakyReLU(alpha=0.2)(d1_2)
    # C128
    d3_2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d2_2)
    d3_2 = BatchNormalization()(d3_2)
    d3_2 = LeakyReLU(alpha=0.2)(d3_2)
    d3_2 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d3_2)
    d3_2 = BatchNormalization()(d3_2)
    d4_2 = LeakyReLU(alpha=0.2)(d3_2)
    # C256
    d5_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d4_2)
    d5_2 = BatchNormalization()(d5_2)
    d5_2 = LeakyReLU(alpha=0.2)(d5_2)
    d5_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d5_2)
    d5_2 = BatchNormalization()(d5_2)
    d6_2 = LeakyReLU(alpha=0.2)(d5_2)
    # C512
    d7_2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d6_2)
    d7_2 = BatchNormalization()(d7_2)
    d7_2 = LeakyReLU(alpha=0.2)(d7_2)
    d7_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d7_2)
    d7_2 = BatchNormalization()(d7_2)
    d8_2 = LeakyReLU(alpha=0.2)(d7_2)

    # bottleneck, no batch norm and relu
    b1_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d8_2)
    b2_2 = Activation('relu')(b1_2)

    # upsampling
    # c512
    u1_2 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(b2_2)
    u2_2 = BatchNormalization()(u1_2)
    u3_2 = Activation('relu')(u2_2)
    # c256
    u4_2 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u3_2)
    u5_2 = BatchNormalization()(u4_2)
    u6_2 = Activation('relu')(u5_2)
    # c128
    u7_2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u6_2)
    u8_2 = BatchNormalization()(u7_2)
    u9_2 = Activation('relu')(u8_2)
    # c64
    u10_2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u9_2)
    u11_2 = BatchNormalization()(u10_2)
    u12_2 = Activation('relu')(u11_2)

    u13_2 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u12_2)
    out_image_2 = Activation('sigmoid')(u13_2)

    # source image input, MODALITY 4
    in_src_image4 = Input(shape=image_shape)
    # C64
    d1_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(in_src_image4)
    d1_3 = BatchNormalization()(d1_3)
    d1_3 = LeakyReLU(alpha=0.2)(d1_3)
    d1_3 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d1_3)
    d1_3 = BatchNormalization()(d1_3)
    d2_3 = LeakyReLU(alpha=0.2)(d1_3)
    # C128
    d3_3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d2_3)
    d3_3 = BatchNormalization()(d3_3)
    d3_3 = LeakyReLU(alpha=0.2)(d3_3)
    d3_3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d3_3)
    d3_3 = BatchNormalization()(d3_3)
    d4_3 = LeakyReLU(alpha=0.2)(d3_3)
    # C256
    d5_3= Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d4_3)
    d5_3 = BatchNormalization()(d5_3)
    d5_3 = LeakyReLU(alpha=0.2)(d5_3)
    d5_3 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d5_3)
    d5_3 = BatchNormalization()(d5_3)
    d6_3 = LeakyReLU(alpha=0.2)(d5_3)
    # C512
    d7_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d6_3)
    d7_3 = BatchNormalization()(d7_3)
    d7_3 = LeakyReLU(alpha=0.2)(d7_3)
    d7_3 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d7_3)
    d7_3 = BatchNormalization()(d7_3)
    d8_3 = LeakyReLU(alpha=0.2)(d7_3)

    # bottleneck, no batch norm and relu
    b1_3 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d8_3)
    b2_3 = Activation('relu')(b1_3)

    # upsampling
    # c512
    u1_3 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(b2_3)
    u2_3 = BatchNormalization()(u1_3)
    u3_3 = Activation('relu')(u2_3)
    # c256
    u4_3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u3_3)
    u5_3 = BatchNormalization()(u4_3)
    u6_3 = Activation('relu')(u5_3)
    # c128
    u7_3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u6_3)
    u8_3 = BatchNormalization()(u7_3)
    u9_3 = Activation('relu')(u8_3)
    # c64
    u10_3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u9_3)
    u11_3 = BatchNormalization()(u10_3)
    u12_3 = Activation('relu')(u11_3)

    u13_3 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u12_3)
    out_image_3 = Activation('sigmoid')(u13_3)

    # source image input, MODALITY 5
    in_src_image5 = Input(shape=image_shape)
    # C64
    d1_4 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(in_src_image5)
    d1_4 = BatchNormalization()(d1_4)
    d1_4 = LeakyReLU(alpha=0.2)(d1_4)
    d1_4 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d1_4)
    d1_4 = BatchNormalization()(d1_4)
    d2_4 = LeakyReLU(alpha=0.2)(d1_4)
    # C128
    d3_4 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d2_4)
    d3_4 = BatchNormalization()(d3_4)
    d3_4 = LeakyReLU(alpha=0.2)(d3_4)
    d3_4 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d3_4)
    d3_4 = BatchNormalization()(d3_4)
    d4_4 = LeakyReLU(alpha=0.2)(d3_4)
    # C256
    d5_4 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d4_4)
    d5_4 = BatchNormalization()(d5_4)
    d5_4 = LeakyReLU(alpha=0.2)(d5_4)
    d5_4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d5_4)
    d5_4 = BatchNormalization()(d5_4)
    d6_4 = LeakyReLU(alpha=0.2)(d5_4)
    # C512
    d7_4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(d6_4)
    d7_4 = BatchNormalization()(d7_4)
    d7_4 = LeakyReLU(alpha=0.2)(d7_4)
    d7_4 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d7_4)
    d7_4 = BatchNormalization()(d7_4)
    d8_4 = LeakyReLU(alpha=0.2)(d7_4)

    # bottleneck, no batch norm and relu
    b1_4 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d8_4)
    b2_4 = Activation('relu')(b1_4)

    # upsampling
    # c512
    u1_4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(b2_4)
    u2_4 = BatchNormalization()(u1_4)
    u3_4 = Activation('relu')(u2_4)
    # c256
    u4_4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u3_4)
    u5_4 = BatchNormalization()(u4_4)
    u6_4 = Activation('relu')(u5_4)
    # c128
    u7_4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u6_4)
    u8_4 = BatchNormalization()(u7_4)
    u9_4 = Activation('relu')(u8_4)
    # c64
    u10_4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u9_4)
    u11_4 = BatchNormalization()(u10_4)
    u12_4 = Activation('relu')(u11_4)

    u13_4 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u12_4)
    out_image_4 = Activation('sigmoid')(u13_4)


    # generation of T2
    g_d1 = MixedFusion_block_0(d2_0, d2_1, d2_2, d2_3, d2_4)
    # g_d1 = BatchNormalization()(g_d1)
    g_d11 = maxpooling()(g_d1)
    g_d2 = MixedFusion_block_d(d4_0, d4_1, d4_2, d4_3, d4_4, g_d11)
    # g_d2 = BatchNormalization()(g_d2)
    g_d21 = maxpooling()(g_d2)
    g_d3 = MixedFusion_block_d(d6_0, d6_1, d6_2, d6_3, d6_4, g_d21)
    # g_d3 = BatchNormalization()(g_d3)
    g_d31 = maxpooling()(g_d3)
    g_d4 = MixedFusion_block_d(d8_0, d8_1, d8_2, d8_3, d8_4, g_d31)

    # bottleneck, no batch norm and relu
    b_s1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(g_d4)
    b_s1 = Activation('relu')(b_s1)
    b_s2 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(b_s1)
    b_s2 = Activation('relu')(b_s2)
    b_s2 = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(b_s2)
    b_s2 = Activation('relu')(b_s2)
    b_s2 = UpSampling2D(size=(2, 2))(b_s2)

    g_u1 = MixedFusion_block_u(u3_0, u3_1, u3_2, u3_3, u3_4, b_s2, g_d4)
    # g_u1 = BatchNormalization()(g_u1)
    g_u1 = UpSampling2D(size=(2, 2))(g_u1)
    g_u2 = MixedFusion_block_u(u6_0, u6_1, u6_2, u6_3, u6_4, g_u1, g_d3)
    # g_u2 = BatchNormalization()(g_u2)
    g_u2 = UpSampling2D(size=(2, 2))(g_u2)
    g_u3 = MixedFusion_block_u(u9_0, u9_1, u9_2, u9_3, u9_4, g_u2, g_d2)
    # g_u3 = BatchNormalization()(g_u3)
    g_u3 = UpSampling2D(size=(2, 2))(g_u3)
    g_u4 = MixedFusion_block_u(u12_0, u12_1, u12_2, u12_3, u12_4, g_u3, g_d1)
    # g_u4 = BatchNormalization()(g_u4)
    g_u4 = UpSampling2D(size=(2, 2))(g_u4)
    g_ue = Conv2D(1, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(g_u4)
    gen_image = Activation('sigmoid')(g_ue)

    model = Model([in_src_image1, in_src_image2, in_src_image3, in_src_image4, in_src_image5], [out_image_0, out_image_1, out_image_2, out_image_3, out_image_4, gen_image])

    return model

def define_gan(g_model, d_model, image_shape):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    in_src1 = Input(shape=image_shape)
    in_src2 = Input(shape=image_shape)
    in_src3 = Input(shape=image_shape)
    in_src4 = Input(shape=image_shape)
    in_src5 = Input(shape=image_shape)
    out_src1, out_src2, out_src3, out_src4, out_src5, gen_out  = g_model([in_src1, in_src2, in_src3, in_src4, in_src5])

    dis_out = d_model(gen_out)
    model = Model([in_src1, in_src2, in_src3, in_src4, in_src5], [dis_out, out_src1, out_src2, out_src3, out_src4, out_src5, gen_out])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae', 'mae', 'mae', 'mae', 'mae', 'mae'], optimizer=opt, loss_weights=LOSSES_WEIGHTS)
    return model
