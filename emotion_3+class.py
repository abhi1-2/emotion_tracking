
# coding: utf-8

# In[1]:

import dlib
import numpy as np
import cv2 as cv
import os

import matplotlib.pyplot as plt
face_detector=dlib.get_frontal_face_detector()
predictor_model='shape_predictor_68_face_landmarks.dat'
predictor=dlib.shape_predictor(predictor_model)
facerec=dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
main_path='/home/ashman/emotion/facial_expressions/dataset/'
d=os.listdir(main_path)
img_array=[]
img_label=[]

count=0
for i,dirs in enumerate(sorted(d)):
    
    print(dirs)
    for item in (os.listdir(main_path+dirs)):
        label=[0,0,0]
        if(item.endswith('.jpg')):
            img=cv.imread(main_path+dirs+'/'+item)
            gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            face_rect=face_detector(gray,1)
            for k,d in enumerate(face_rect):
                shape=predictor(gray,d)
                face_descriptor=facerec.compute_face_descriptor(img,shape)
            label[i]=1
            img_array.append(face_descriptor)
            img_label.append(label)
            
            count+=1
            if(count>249):
                break

    count=0


# In[2]:

import numpy as np
img_dataset=np.array(img_array)
img_labels=np.array(img_label)
print((img_dataset.shape))
print((img_labels.shape))


# In[3]:

#img_labels[89]


# In[4]:

img_size=128
def reformat(dataset):
    dataset=dataset.reshape(-1,1*img_size).astype(np.float32)
    return dataset
train_dataset=reformat(img_dataset)
train_labels=img_labels
print(train_labels.shape)


# In[5]:

train_labels[1]


# In[6]:

#tensorflow model
import tensorflow as tf
#train_subset=105
graph=tf.Graph()
hidden_nodes1=80
hidden_nodes2=64

#patch_size=5
batch_size=10
image_size=128
num_classes=train_labels.shape[1]
depth=10
with graph.as_default():
    tf_train_dataset=tf.placeholder(tf.float32,shape=(batch_size,1*image_size))
    tf_train_labels=tf.placeholder(tf.float32,shape=(batch_size,num_classes))
    
    
    #tf_valid_dataset=tf.constant(valid_dataset)
    #tf_test_dataset=tf.constant(test_dataset)
    
    weights_l1=tf.Variable(tf.truncated_normal([1*image_size,hidden_nodes1]))
    biases_l1=tf.Variable(tf.zeros([hidden_nodes1]))
    weights_l2=tf.Variable(tf.truncated_normal([hidden_nodes1,hidden_nodes2]))
    biases_l2=tf.Variable(tf.zeros([hidden_nodes2]))
    weights_l3=tf.Variable(tf.truncated_normal([hidden_nodes2,num_classes]))
    biases_l3=tf.Variable(tf.zeros([num_classes]))
    global_step=tf.Variable(0)
    learning_rate =tf.train.exponential_decay(0.06, global_step,100, 0.9, staircase=True)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)#
    
 #graph structure   
    def computation(dataset):
        logits_1=tf.matmul(dataset,weights_l1)+biases_l1
        logits_1=tf.nn.relu(logits_1)
        #keep_prob1=tf.placeholder(tf.float32)
        #drop_out=tf.nn.dropout(logits_1,keep_prob1)
        logits_2=tf.matmul(logits_1,weights_l2)+biases_l2
       # keep_prob=tf.placeholder(tf.float32)
        #drop_out=tf.nn.dropout(logits_2,keep_prob)
        logits_2=tf.nn.relu(logits_2)
        logits_3=tf.matmul(logits_2,weights_l3)+biases_l3
        
        return logits_3
        
        
        
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=computation(tf_train_dataset),labels=tf_train_labels)+0.01*(tf.nn.l2_loss(weights_l1)+tf.nn.l2_loss(biases_l1)+tf.nn.l2_loss(weights_l2)
                                                                                                                              +tf.nn.l2_loss(biases_l2)+tf.nn.l2_loss(weights_l3)+tf.nn.l2_loss(biases_l3)))          
    
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)#,global_step=global_step)
    #optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)#,global_step=global_step)
    train_prediction=tf.nn.softmax(computation(tf_train_dataset))
    



# In[7]:

num_steps=201
alpha=['anger','happiness','sadness']
def accuracy(prediction,labels):
    return (100.0*np.sum(np.argmax(prediction,1)==np.argmax(labels,1)))/prediction.shape[0]

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('initialized')
    
    for step in range(num_steps):
        offset=(step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_dataset=train_dataset[offset:(offset+batch_size),:]
        batch_labels=train_labels[offset:(offset+batch_size),:]
        feed_dict={tf_train_dataset:batch_dataset, tf_train_labels:batch_labels}
       
    
        _,l,p=session.run([optimizer,loss,train_prediction,],feed_dict=feed_dict)
        if(step%50==0):
            print("loss at ",step,':',l)
            print('training accuracy :',accuracy(p,batch_labels))
            img=cv.imread('/home/ashman/Downloads/smile-300x300.jpg' )
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            detected_faces = face_detector(gray, 1)
            for k,d in enumerate(detected_faces):
                shape=predictor(gray,d)
                face_descriptor = facerec.compute_face_descriptor(img,shape )
                image_data1=np.array(face_descriptor,dtype=np.float32)
                img=image_data1.reshape(1,1*128)
                print(tf.nn.softmax(computation(img)).eval())
                test=tf.nn.softmax(computation(img)).eval()
                index=np.argmax(test)
                print("predicted class is: ",alpha[index])    
    img=cv.imread('/home/ashman/Downloads/smile-300x300.jpg' )
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detected_faces = face_detector(gray, 1)
    for k,d in enumerate(detected_faces):
        shape=predictor(gray,d)
        face_descriptor = facerec.compute_face_descriptor(img,shape )
        image_data1=np.array(face_descriptor,dtype=np.float32)
        img=image_data1.reshape(1,1*128)
        print(tf.nn.softmax(computation(img)).eval())
        test=tf.nn.softmax(computation(img)).eval()
        index=np.argmax(test)
        print("predicted class is: ",alpha[index]) 
    
    saver_path=saver.save(session,'/home/ashman/emotion/facial_expressions/model.ckpt')
    print('saved')
# In[ ]:




# In[ ]:




# In[ ]:



