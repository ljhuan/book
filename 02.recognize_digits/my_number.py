import os
import paddle.v2 as paddle
import numpy as np
from PIL import Image

def convolutional_neural_network(img):
    # first conv layer
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # second conv layer
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # fully-connected layer
    predict = paddle.layer.fc(
        input=conv_pool_2, size=10, act=paddle.activation.Softmax())
    return predict

def load_image(file):
	im = Image.open(file).convert('L')
	im = im.resize((28, 28), Image.ANTIALIAS)
	im = np.array(im).astype(np.float32).flatten()
	im = im / 255.0 * 2.0 - 1.0
	return im


def main():
	# Initialize PaddlePaddle.
	paddle.init(use_gpu=False, trainer_count=1)

	# Configure the neural network.
	images = paddle.layer.data(name='pixel', type=paddle.data_type.dense_vector(784))
	predict = convolutional_neural_network(images)

	with open('params_pass_3.tar', 'r') as f:
	    parameters = paddle.parameters.Parameters.from_tar(f)

	test_data = []
	cur_dir = os.path.dirname(os.path.realpath(__file__))
	test_data.append((load_image(cur_dir + '/image/3.png'), ))

	# Infer using provided test data.
	probs = paddle.infer(
	     output_layer=predict, parameters=parameters, input=test_data)

	lab = np.argsort(-probs)  # probs and lab are the results of one batch data
	print "Label of image/3.png is: %d" % lab[0][0]

if __name__ == '__main__':
    main()
