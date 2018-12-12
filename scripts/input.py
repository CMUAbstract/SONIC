import argparse
import pickle
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def main(args):
	input = read_data_sets(args.data_dir)
	data, label = input.train.next_batch(1)
	print('Label: %d' % label.tolist()[0])
	with open(args.dest, 'w+') as f:
		pickle.dump(data, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data_dir',
		type=str,
		help='Data director')
	parser.add_argument(
		'--dest',
		type=str,
		help='Destination file')
	args = parser.parse_args()
	main(args)