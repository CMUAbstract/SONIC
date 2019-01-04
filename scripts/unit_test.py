import os
import pickle
import numpy as np
from scipy import signal
from scipy import sparse

# run as python scripts/unit_test.py

def apply_f_lit(a):
	return 'F_LIT(' + str(a) + ')'

def writeHeader(name, dest, args):
	f = open(os.path.join(dest, name + '.h'), 'w+')
	str_var = '#ifndef ' + name.upper() + '_H\n'
	str_var += '#define ' + name.upper() + '_H\n'
	str_var += '#include \'<libfixed/fixed.h>\'\n'
	str_var += '#include \'<libdnn/mem.h>\'\n\n'
	for arg_name, arg, form in args:
		if form == 'dense': # Something here!!!!
			# print 'Dense'
			values = np.squeeze(arg).astype(dtype=str)
			shape = values.shape
			flit = np.vectorize(apply_f_lit)
			values = flit(values).tolist()
			str_arr = str(values).replace('[', '{').replace(']', '}')
			str_dim = ''
			for dim in shape:
				str_dim += '[' + str(dim) + ']' 

			str_var += '__ro_hifram fixed ' + arg_name + \
				str_dim + ' = ' + str_arr + ';\n\n'
			
		elif len(arg.shape) > 3: # High Dimensional Sparse
			# print 'High-Dimensional Sparse'
			str_arr = ''
			str_diff = ''
			str_size = ''
			size = 0
			values = arg.reshape(arg.shape[0], -1)
			for v in values:
				value = v[v != 0.0].astype(dtype=str)

				idx = np.where(v != 0.0)[0]
				diff = np.diff(idx).astype(dtype=str).tolist()
				if value.shape[0] > 0:
					diff.insert(0, str(idx[0]))

					flit = np.vectorize(apply_f_lit)
					value = flit(value).tolist()
					str_arr += str(value).replace('[', '').replace(']', '') + ','
					str_diff += str(diff).replace('[', '').replace(']', '') + ','
					str_size += str(len(value)) + ','
					size += len(value)
				else:
					str_size += '0,'

			str_arr = str_arr[:-1]
			str_diff = str_diff[:-1]
			str_size = str_size[:-1]
			shape = values.shape
			str_var += '#define ' + arg_name.upper() + '_LEN ' + str(size) + '\n\n'
			str_var += '__ro_hifram fixed ' + arg_name + \
				'[' + str(size) + '] = {' + str_arr + '};\n\n'
			str_var += '__ro_hifram fixed ' + arg_name + '_offsets[' + \
				str(size) + '] = {' + str_diff + '};\n\n'
			str_var += '__ro_hifram fixed ' + arg_name + '_sizes[' + \
				str(shape[0]) + '] = {' + str_size + '};\n\n'
		else: # Low Dimensional Sparse
			# print "Low Dimensional Sparse"
			csr = sparse.csr_matrix(arg)
			print arg.shape
			data, indices, indptr = csr.data, csr.indices, csr.indptr

			flit = np.vectorize(apply_f_lit)
			data = flit(data).tolist()
			str_arr = str(data).replace('[', '{').replace(']', '}')
			str_indices = str(indices.tolist()).replace('[', '{').replace(']', '}')
			str_indptr = str(indptr.tolist()).replace('[', '{').replace(']', '}')

			str_var += '#define ' + arg_name.upper() + '_LEN ' + str(len(data)) + '\n\n'
			str_var += '__ro_hifram fixed ' + arg_name + '[' + \
				str(len(data)) + '] = ' + str_arr + ';\n\n'

			str_var += '__ro_hifram uint16_t ' + arg_name + '_offsets[' + \
				str(len(indices)) + '] = ' + str_indices + ';\n\n'

			str_var += '__ro_hifram uint16_t ' + arg_name + '_sizes[' + \
				str(len(indptr)) + '] = ' + str_indptr + ';\n\n'

	str_var = str_var.replace("'", '')
	str_var += '#endif'
	f.write(str_var)
	f.close() 

data = {'a': (15, 10), 'b': (10, 1), 'c': (3, 1, 3, 3)}
# data = {'a': (3, 3), 'b': (3, 1), 'c': (3, 3, 3)}
def conv(a, b, mode='same'):
	stack = []
	for i in xrange(b.shape[0]):
		t = a
		if mode == 'same':
			rows, cols = a.shape
			_, flayers, frows, fcols = b.shape
			t = np.zeros([rows + (frows - 1), cols + (fcols - 1)])
			t[:a.shape[0], :a.shape[1]] = a
		stack.append(signal.correlate2d(t, np.squeeze(b[i]), mode='valid'))

	return np.stack(stack, axis=0).astype(int)

def convT(a, b, mode='same'):
	a = a.T
	b = np.transpose(b, (0, 1, 3, 2))
	return np.transpose(conv(a, b, mode), (0, 2, 1))

tests = {
		'dm_mul/sm_mul/svm_mul': lambda m, v, _: m.dot(v),
		'dm_conv/sm_conv - same': lambda m, _, f: conv(m, f, 'same'),
		'dm_conv/sm_conv': lambda m, _, f: conv(m, f, 'valid'),
		'dm_conv/sm_conv - transposed': lambda m, _, f: convT(m, f, 'valid'),
		'dm_conv/sm_conv - stride': lambda m, _, f: conv(m, f, 'valid')[:,::2,::2]
		}

def main():
	for d in data:
		path = os.path.join('params/test', d + '.param')
		if not os.path.exists(path):
			data[d] = np.random.randint(-5, 5, size=data[d])
			data[d][data[d] == 1] = 0
			data[d][data[d] == -1] = 0
			print d, data[d].shape
			writeHeader(d + '_dense', 'headers', [(d + '_dense', data[d], 'dense')])
			writeHeader(d + '_sparse', 'headers', [(d + '_sparse', data[d], '')])
			with open(path, 'w+') as f:
				pickle.dump(data[d], f)
		else:
			with open(path, 'r') as f:
				data[d] = pickle.load(f)

	print 'a'
	print data['a']
	print 'b'
	print data['b']
	print 'c'
	print data['c']
	for t in tests:
		print t
		print tests[t](data['a'], data['b'], data['c'])

if __name__ == '__main__':
	main()