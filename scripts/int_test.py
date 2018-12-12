import os
import argparse
import pickle
import numpy as np
from scipy import ndimage
from scipy import signal

file_path = None

class Graph:
	class Node:
		def __init__(self, name, op, a, b):
			self.name = name
			self.op = op
			self.a = a
			self.b = b

	def __init__(self):
		self.nodes = {}

	def append(self, name, op, a=None, b=None):
		if type(a) is str:
			a = self.nodes[a]
		if type(b) is str:
			b = self.nodes[b]
		self.nodes[name] = Graph.Node(name, op, a, b)
	
	def eval(self, ref):
		node = ref
		if type(ref) is str:
			node = self.nodes[ref]

		if node.a is None and node.b is None:
			return node.op(node.name)
		elif node.b is None:
			return node.op(node.name, self.eval(node.a))
		else:
			return node.op(node.name, self.eval(node.a), self.eval(node.b))

def weight(name):
	global file_path
	path = os.path.join(file_path, name + '.param')
	with open(path, 'r') as f:
		return pickle.load(f)

def relu(name, a):
	return np.fabs(np.multiply(a, a > 0))

def flatten(name, a):
	flattened = a.flatten()
	rows = flattened.shape[0]
	return np.reshape(flattened, [rows, 1])

def shuffle_indices(layers, rows, cols):
		size = layers * rows * cols
		layer_size = rows * cols
		idx = 0
		offset = 0
		count = 0
		permutation = [0] * size
		while count != size:
			if idx > size - layers:
				offset += 1
				idx = 0
			permutation[offset + idx] = count
			idx += layers
			count += 1

		return permutation

def shuffle(name, a):
	permutation = np.argsort(shuffle_indices(100, 4, 4))
	return a[:,permutation]

def mul(name, a, b):
	return np.multiply(a, b)

def add(name, a, b):
	if len(b.shape) == 1:
		b = np.reshape(b, [b.shape[0], 1])
	return np.add(a, b)

def conv_add(name, a, b):
	layers = a.shape[0]
	stack = []
	for l in xrange(layers):
		stack.append(np.add(a[l], b[l]))

	return np.stack(stack, axis=0)

def mmul(name, a, b):
	return np.dot(a, b)

def conv(name, a, b):
	filter = a
	layers = filter.shape[0]
	stack = []
	distrib = False
	if len(filter.shape) == len(b.shape): distrib = True
	for l in xrange(layers):
		if distrib:
			stack.append(signal.correlate(b[l], filter[l], mode='valid'))
		else:
			stack.append(signal.correlate(b, filter[l], mode='valid'))
	return np.stack(stack, axis=0)

def rand_matrix(shape):
	return np.rand_matrix(-5, 5, size=shape)

def pooling(mat, ksize, method='max', pad=False):
	m, n = mat.shape[:2]
	ky,kx = ksize

	_ceil = lambda x, y: int(np.ceil(x/float(y)))

	if pad:
		ny = _ceil(m, ky)
		nx = _ceil(n, kx)
		size = (ny * ky, nx * kx) + mat.shape[2:]
		mat_pad = np.full(size, np.nan)
		mat_pad[:m,:n,...] = mat
	else:
		ny = m // ky
		nx = n // kx
		mat_pad = mat[:ny * ky, :nx * kx, ...]

	new_shape=(ny, ky, nx, kx) + mat.shape[2:]

	if method == 'max':
		result = np.nanmax(mat_pad.reshape(new_shape), axis=(1,3))
	else:
		result = np.nanmean(mat_pad.reshape(new_shape), axis=(1,3))

	return result

def maxpool2x2(name, a):
	layers = a.shape[0]
	stacks = []
	for l in xrange(layers):
		stacks.append(pooling(np.squeeze(a[l]), (2, 2)))
	return np.stack(stacks, axis=0)

def transpose(name, a):
	return a.T

def permute(name, a):
	return np.transpose(a, axes=[3, 2, 0, 1])

def permute_vh(name, a):
	return np.transpose(a, axes=[2, 3, 0, 1])

def squeeze(name, a):
	return np.squeeze(a)

def input_reshape(name, a):
	return np.reshape(a, [1, 28, 28])

def arg_max(name, a):
	return np.argmax(a.flatten())

# test_data = {'a': (15, 10), 'b': (10, 1), 'c': (3, 1, 3, 3)}

def main(args):
	global file_path
	file_path = args.src_dir

	'''
	for d in test_data:
		path = os.path.join('params/test', d + '.param')
		if not os.path.exists(path):
			test_data[d] = np.random.randint(-5, 5, size=data[d])
			with open(path, 'w+') as f:
				pickle.dump(mat, f)
		else:
			with open(path, 'r') as f:
				test_data[d] = pickle.load(f)

	print 'a', test_data['a']
	print 'b', test_data['b']
	print 'c', test_data['c']
	conv_result = conv('', test_data['c'], test_data['a'])
	print conv_result
	print maxpool2x2('', conv_result)
	return
	'''

	graph = Graph()
	graph.append('input', weight)
	graph.append('input_reshape', input_reshape, 'input')

	graph.append('conv1_wd', weight)
	graph.append('conv1_md', weight)
	graph.append('conv1_wh', weight)
	graph.append('conv1_mh', weight)
	graph.append('conv1_wv', weight)
	graph.append('conv1_mv', weight)
	graph.append('conv1_b', weight)
	graph.append('conv1_wmd_', mul, 'conv1_wd', 'conv1_md')
	graph.append('conv1_wmd', permute, 'conv1_wmd_')

	graph.append('conv1_wmh_', mul, 'conv1_wh', 'conv1_mh')
	graph.append('conv1_wmh', permute_vh, 'conv1_wmh_')
	graph.append('conv1_wmv_', mul, 'conv1_wv', 'conv1_mv')
	graph.append('conv1_wmv', permute_vh, 'conv1_wmv_')

	graph.append('conv2_w', weight)
	graph.append('conv2_m', weight)
	graph.append('conv2_mw_', mul, 'conv2_w', 'conv2_m')
	graph.append('conv2_mw', permute, 'conv2_mw_')
	graph.append('conv2_b', weight)

	graph.append('fc1_wh', weight)
	graph.append('fc1_mh', weight)
	graph.append('fc1_wv', weight)
	graph.append('fc1_mv', weight)
	graph.append('fc1_b', weight)
	graph.append('fc1_wmh__', mul, 'fc1_wh', 'fc1_mh')
	graph.append('fc1_wmh_', transpose, 'fc1_wmh__')
	graph.append('fc1_wmh', shuffle, 'fc1_wmh_')
	graph.append('fc1_wmv_', mul, 'fc1_wv', 'fc1_mv')
	graph.append('fc1_wmv', transpose, 'fc1_wmv_')

	graph.append('fc2_w', weight)
	graph.append('fc2_wt', transpose, 'fc2_w')
	graph.append('fc2_b', weight)

	graph.append('conv1_d', conv, 'conv1_wmd', 'input_reshape')
	graph.append('conv1_h', conv, 'conv1_wmh', 'conv1_d')
	graph.append('conv1_v', conv, 'conv1_wmv', 'conv1_h')
	graph.append('conv1', conv_add, 'conv1_v', 'conv1_b')
	graph.append('conv1r', relu, 'conv1')
	graph.append('conv1squeeze', squeeze, 'conv1r')
	graph.append('conv1max', maxpool2x2, 'conv1squeeze')

	graph.append('conv2', conv, 'conv2_mw', 'conv1max')
	graph.append('conv2b', conv_add, 'conv2', 'conv2_b')
	graph.append('conv2r', relu, 'conv2b')
	graph.append('conv2max', maxpool2x2, 'conv2r')
	graph.append('conv2flat', flatten, 'conv2max')

	graph.append('fc1_h', mmul, 'fc1_wmh', 'conv2flat')
	graph.append('fc1_v', mmul, 'fc1_wmv', 'fc1_h')
	graph.append('fc1', add, 'fc1_v', 'fc1_b')
	graph.append('fc1r', relu, 'fc1')

	graph.append('fc2', mmul, 'fc2_wt', 'fc1r')
	graph.append('fc2b', add, 'fc2', 'fc2_b')
	graph.append('predict', arg_max, 'fc2b')

	np.set_printoptions(precision=3, linewidth=200, suppress=True)

	print graph.eval('fc2b').flatten().tolist()
	print('Prediction: %d' % graph.eval('predict'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--src_dir',
		type=str,
		help='Source directory')
	args = parser.parse_args()
	main(args)