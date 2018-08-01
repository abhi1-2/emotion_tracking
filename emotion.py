
# coding: utf-8

# In[12]:



# In[1]:

import dlib
import numpy as np
import cv2 as cv
import os
import tensorflow as tf
face_detector=dlib.get_frontal_face_detector()
predictor_model='/Users/avisheksarkar/Downloads/shape_predictor_68_face_landmarks.dat'
predictor=dlib.shape_predictor(predictor_model)
facerec=dlib.face_recognition_model_v1('/Users/avisheksarkar/Downloads/dlib_face_recognition_resnet_model_v1.dat')



# In[ ]:



#img=cv.imread('/Users/avisheksarkar/Desktop/new1/dataSet/User1/User1.30.jpg' )
cap = cv.VideoCapture(0)
ret,frame=cap.read()
while ret:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detected_faces = face_detector(gray, 1)
    for k,d in enumerate(detected_faces):
        shape=predictor(gray,d)
        face_descriptor = facerec.compute_face_descriptor(img,shape )
        image_data1=np.array(face_descriptor,dtype=np.float32)
        img=image_data1.reshape(-1,1*128)
        print(img.shape)
        saver=tf.train.import_meta_graph('/Users/avisheksarkar/Desktop/emotion/facial_expressions/model.ckpt.meta')
        graph=tf.get_default_graph()
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            saver.restore(session,'/Users/avisheksarkar/Desktop/emotion/facial_expressions/model.ckpt')
            logits_1=tf.matmul(img,tf.trainable_variables()[0])+tf.trainable_variables()[1]
            logits_1=tf.nn.relu(logits_1)
            #keep_prob1=tf.placeholder(tf.float32)
            #drop_out=tf.nn.dropout(logits_1,keep_prob1)
            logits_2=tf.matmul(logits_1,tf.trainable_variables()[2])+tf.trainable_variables()[3]
           #keep_prob=tf.placeholder(tf.float32)
            #drop_out=tf.nn.dropout(logits_2,keep_prob)
            logits_2=tf.nn.relu(logits_2)
            logits_3=tf.matmul(logits_2,tf.trainable_variables()[4])+tf.trainable_variables()[5]
            result=tf.nn.softmax(logits_3)
            print(result.eval())


# In[ ]:



