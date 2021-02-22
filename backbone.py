import tensorflow as tf
import common

def darknet53(input_data):

    input_data = common.conv2d_BN_Leaky(input_data, (32, 3, 3, 3))
    input_data = common.conv2d_BN_Leaky(input_data, (64, 3, 3, 32), downsample=True)

    for i in range(1):
        input_data = common.residual_block(input_data, 32, 64, 64)
    
    input_data = common.conv2d_BN_Leaky(input_data, (128, 3, 3, 64), downsample=True)

    for i in range(2):
        input_data = common.residual_block(input_data, 64, 128, 128)
    
    input_data = common.conv2d_BN_Leaky(input_data, (256, 3, 3, 128), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 128, 256, 256)

    output_1 = input_data
    input_data = common.conv2d_BN_Leaky(input_data, (512, 3, 3, 256), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 256, 512, 512)

    output_2 = input_data
    input_data = common.conv2d_BN_Leaky(input_data, (1024, 3, 3, 512), downsample=True)

    for i in range(4):
        input_data = common.residual_block(input_data, 512, 1024, 1024)

    output_3 = input_data

    return output_1, output_2, output_3
    

def darknet53_classifier():
    
    inputs = tf.keras.Input(shape=(416, 416, 3))
    _, _, x = darknet53(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1000, activation='softmax')(x)

    model = tf.keras.Model(inputs, x)
    return model

    
if __name__ == '__main__':
    model = darknet53_classifier()
    model.summary()
