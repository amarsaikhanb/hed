import numpy as np
from numpy import array
import cv2

cap = cv2.VideoCapture(0)
capture = None
while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (256, 256))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	capture = frame
    	break

cap.release()
cv2.destroyAllWindows()


import tensorflow as tf
from tensorflow.python.platform import gfile
import os
# output_node_names = 'hed/dsn_fuse/conv2d/BiasAdd:0'
# input_node_names = 'hed_input:0'
# nod_name = 'save/Const:0'

GRAPH_PB_PATH = 'hed.pb'
print('isfile', os.path.isfile(GRAPH_PB_PATH))
print('capture.shape', capture.shape)

# kInputLayerName = tf.Tensor("hed_input:0", shape=(1, 256, 256, 3), dtype=float)
# kIsTrainingName = tf.Tensor("is_training:0", dtype=bool)
# kOutputLayerName =  tf.Tensor("hed/dsn_fuse/conv2d/BiasAdd:0", shape=(1, 256, 256, 1), dtype=float)
           

kInputLayerName = "hed_input"
kIsTrainingName = "is_training"
kOutputLayerName = "hed/dsn_fuse/conv2d/BiasAdd"
with tf.Graph().as_default() as graph: 
	with tf.Session() as sess:
		print("load graph")
		tf.initializers.global_variables()
		with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
			print("Load Image...")
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			sess.graph.as_default()
			tf.import_graph_def(graph_def)
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()
			input_tensor =  array(capture).reshape(1, 256, 256, 3)
			is_training = False
			# for op in sess.graph.get_operations():
			# 	print(op)
			#finalOutput = tf.Tensor('outputs:0', shape=(1, 256, 256, 3), dtype='float32')
			#print('run_options', input_tensor)
			# feed_dict = {
			# 	'hed_input':'0',
			# 	'is_training':'0'
			# }
			sess.run(run_options, input_tensor,feed_dict = {'hed_input':0,'is_training':False})
			#, {'hed/dsn_fuse/conv2d/BiasAdd'})#feed_dict={kInputLayerName: input_tensor, kIsTrainingName: is_training})#,{kOutputLayerName}, {}, run_metadata)\
			#self.outputs = 'Tensor('outputs:0', shape=(128, 250, 250, 3), dtype=float32)'






