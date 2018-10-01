import tensorflow as tf
import imutils
import numpy as np
import argparse
import facenet1
import os
import sys
import math
import cv2
import pickle
from sklearn.svm import SVC
from scipy import misc
import align.detect_face
from six.moves import xrange
import mxnet as mx
from mtcnn_detector import MtcnnDetector

#"rtsp://admin:admin0864@103.60.63.138:8081/cam/realmonitor?channel=1&subtype=1"

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
camera = cv2.VideoCapture("id2.mp4")



font = cv2.FONT_HERSHEY_SIMPLEX

#capture the whole frame

def grabVideoFeed():
    grabbed, frame = camera.read()
    return frame if grabbed else None

def initialSetup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



with tf.Graph().as_default():
    with tf.Session() as sess:
    	#path for modeldir

        facenet1.load_model("modeldir")

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        
        sampleNum = 0
        while True:
            frame = grabVideoFeed()
            

            if frame is None:
                raise SystemError('issue in grabbing frame')

   #crop th faces
            results = detector.detect_face(frame)
            if results is None:
                continue

            total_boxes = results[0]
            points = results[1]
            text = "hi_kato"
            for b in total_boxes:
                x1 = int(b[0])
                y1 = int(b[1])
                x2 = int(b[2])
                y2 = int(b[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255))
                # cropping the face from whole frame
                img = frame[y1:y2,x1:x2]
                aligned = misc.imresize(img, (160, 160), interp='bilinear')
                sampleNum = sampleNum+1
                
                # changing the img size(1,160,160,3)

                # print(aligned.shape)
                aligned.shape=(1,aligned.shape[0],aligned.shape[1],aligned.shape[2])
                # print(aligned.shape)
            
            # feeding the image to the network
            
                feed_dict = { images_placeholder: aligned , phase_train_placeholder:False}
                emb = sess.run(embeddings, feed_dict=feed_dict)
                classifier_filename_exp = os.path.expanduser("kato.pkl")
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                predictions = model.predict_proba(emb)
                best_class_indices = np.argmax(predictions, axis=1) 
                x23=str(class_names[best_class_indices[0]])
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                prob = float(best_class_probabilities[0])
                x24 = "{:.2f}".format(prob)
                if x23 == "id1":

                    x23 = "JANAS"

                elif x23 == "id2":

                    x23 = "RAMESH"

                elif x23 == "id3":

                    x23 = "LAKS"

                elif x23 =="id4":

                    x23 ="SURESH"

                
                cv2.putText(frame, x23, (x1,y1-10), font,1,(0,0,255),2)
                cv2.putText(frame, x24, (x2,y2+10), font,1,(0,0,255),2)
            
                cv2.imshow("KATOMARAN-LIVE-STREAM",frame)
                
                k=0


                if class_names[best_class_indices[k]] == "id1":
                    cv2.imwrite("/home/sai/kato/prediction/dataset/id1/User."+str(id)+"."+str(sampleNum)+".jpg",aligned)

                elif class_names[best_class_indices[k]] == "id2":
                    cv2.imwrite("/home/sai/kato/prediction/dataset/id2/User."+str(id)+"."+str(sampleNum)+".jpg",aligned)

                elif class_names[best_class_indices[k]] == "id3":
                    cv2.imwrite("/home/sai/kato/prediction/dataset/id3/User."+str(id)+"."+str(sampleNum)+".jpg",aligned)

                elif class_names[best_class_indices[k]] == "id4":
                    cv2.imwrite("/home/sai/kato/prediction/dataset/id4/User."+str(id)+"."+str(sampleNum)+".jpg",aligned)

                else:
                    print("unknown")

                k = cv2.waitKey(1)
                if k == ord('x'):
                    break

        camera.release()
        cv2.destroyAllWindows()