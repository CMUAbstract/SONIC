import os
import pickle
import argparse
import numpy as np
import tensorflow as tf

INPUT_SIZE = 28

param_dir = None
def get_path(file):
	global param_dir
	return os.path.join(param_dir, file)

def main(args):
	global param_dir
	param_dir = args.param_dir
	vars = {
			'input': None, 'conv1_wd': None, 'conv1_md': None, 'conv1_wh': None, 
			'conv1_mh': None, 'conv1_wv': None, 'conv1_mv': None, 'conv1_b': None,
			'conv2_w': None, 'conv2_m': None, 'conv2_b': None, 'fc1_wh': None,
			'fc1_mh': None, 'fc1_wv': None, 'fc1_mv': None, 'fc1_b': None,
			'fc2_w': None, 'fc2_b': None
			}
	for var in vars:
		with open(get_path(var + '.param'), 'r') as f:
			vars[var] = pickle.load(f)

	net = {}
	for var in vars:
		net[var] = tf.get_variable(var, vars[var].shape, 
			initializer=tf.constant_initializer(vars[var]))

	net['input_reshape'] = tf.reshape(net['input'], [-1, INPUT_SIZE, INPUT_SIZE, 1]) 
	net['conv1_wmd'] = tf.multiply(net['conv1_md'], net['conv1_wd'])
	net['conv1_wmh'] = tf.multiply(net['conv1_mh'], net['conv1_wh'])
	net['conv1_wmv'] = tf.multiply(net['conv1_mv'], net['conv1_wv'])
	net['conv2_wm'] = tf.multiply(net['conv2_m'], net['conv2_w'])
	net['fc1_wmh'] = tf.multiply(net['fc1_mh'], net['fc1_wh'])
	net['fc1_wmv'] = tf.multiply(net['fc1_mv'], net['fc1_wv'])

	net['conv1d'] = tf.nn.conv2d(net['input_reshape'], net['conv1_wmd'], 
			[1, 1, 1, 1], padding='VALID')
	net['conv1h'] = tf.nn.depthwise_conv2d(net['conv1d'], net['conv1_wmh'],
		[1, 1, 1, 1], padding='VALID')
	net['conv1v'] = tf.nn.depthwise_conv2d(net['conv1h'], net['conv1_wmv'],
		[1, 1, 1, 1], padding='VALID')
	net['conv1r'] = tf.nn.relu(net['conv1v'] + net['conv1_b'])
	net['conv1max'] = tf.nn.max_pool(net['conv1r'], 
			ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	net['conv2'] = tf.nn.conv2d(net['conv1max'], net['conv2_wm'],
		[1, 1, 1, 1], padding='VALID')
	net['conv2r'] = tf.nn.relu(net['conv2'] + net['conv2_b'])
	net['conv2max'] = tf.nn.max_pool(net['conv2r'], 
			ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	net['conv2flat'] = tf.reshape(net['conv2max'], [-1, 1600])
	net['fc1_h'] = tf.matmul(net['conv2flat'], net['fc1_wmh'])
	net['fc1_v'] = tf.matmul(net['fc1_h'], net['fc1_wmv'])
	net['fc1r'] = tf.nn.relu(net['fc1_v'] + net['fc1_b'])
	net['fc2'] = tf.matmul(net['fc1r'], net['fc2_w']) + net['fc2_b']

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		np.set_printoptions(precision=3, linewidth=200, suppress=True)
		# data = sess.run(net['conv2flat']).flatten().tolist()
		# str_data = ''
		# for i, d in enumerate(data):
		# 	str_data += str(round(d, 2)) + '	'
		# 	if (i + 1) % 25 == 0: 
		# 		print str_data
		# 		str_data = ''
		print sess.run(net['fc2']).flatten().tolist()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--param_dir',
		type=str,
		help='Parameter directory')
	args = parser.parse_args()
	main(args)