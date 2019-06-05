import os
import cv2
from imutils.video import FPS
import imutils
import numpy as np 
import tensorflow as tf 
from tensorflow.python.platform import gfile


cap = cv2.VideoCapture(0)
frameG = None
print('fps:', cap.get(cv2.CAP_PROP_FPS))
fps = FPS().start()
print('live started:')
while(True):
    ret, frame = cap.read()
    #frame = imutils.resize(frame, width=255, height=255)
    frame = cv2.resize(frame, (255, 255))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame_float = tf.to_float(gray)
    #frame_float = frame_float/255.0

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        frameG = frame
        #cv2.imwrite('frameG.jpg', frameG)
        break
fps.stop()
cap.release()
cv2.destroyAllWindows()

output_layer ='hed/dsn_fuse/conv2d/Conv2D'
input_node = 'Placeholder:0'

global_init = tf.global_variables_initializer()

print('load pb file:')
graph_pb = 'hed_graph.pb' 
labels = 'test.txt'
with tf.Session() as sess:
    print('load graph')
    with gfile.FastGFile(graph_pb, 'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes = [n for n in graph_def.node]
    names = []
    for t in graph_nodes:
        names.append(t.name)
    #print(graph_def)
    #sess.run(tf.convert_to_tensor(frameG))
    cv2.imwrite('res.jpg', sess.run(tf.convert_to_tensor(frameG)))
    run_inference_on_image()









#with tf.Session() as sess:
#    sess.run(tf.convert_to_tensor(frameG))
    #cv2.imwrite('res.jpg', sess.run(tf.convert_to_tensor(frameG)))
    #fr = (sess.run(frameG))
    #frame_float = tf.to_float(fr)
    #frame_float = frame_float/255.0
    #cv2.imwrite('res.jpg', frame_float)
    #     sess.run
    #     prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    #     prediction, = sess.run(prob_tensor, {'node': [frameG]})
    # except KeyError:
    #     print('Could not find classification output layer')
    #     print('Verify this a model exported from an Object Detection project')
    #     exit(-1)
