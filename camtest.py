import tensorflow as tf
inputGraph = tf.GraphDef()
import os
#print(os.listdir())
with tf.gfile.Open('hed.pb', 'rb') as f:
	data2read = f.read()
	#print(data2read)
	it = inputGraph.ParseFromString(data2read)
	print(it)