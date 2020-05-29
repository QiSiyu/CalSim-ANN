#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This script runs on a local computer, not Google Colab """

""" ############# User Settings ############# """

""" 1. Select parameters to be used """
input_var = ['SAC','Exp','SJR','DICU','Vern','SF_Tide','DXC']

""" 2. Select stations to be predicted """
""" choose one or more of 'Emmaton','Jersey Point','Collinsville','Rock Slough' """
output_stations=['Jersey Point']

""" 3. Specify folder storing trained models """
model_dir = '/Users/siyuqi/Documents/PhD/3_DSM2/Deliverables'

""" 4. Specify full directory of dataset, including name """
data_dir = '/Users/siyuqi/Documents/PhD/3_DSM2/Data_Code/ANN_data.xlsx'

""" 5. Specify the folder to store computed results """
results_dir = 'predict_results'

""" ########## User Settings End ############ """


from ann_helper import read_data,normalize_in,writeF90
import tensorflow as tf
import os
import numpy as np
import pandas as pd

on_server = False
output_shape = 1

nn_shape = [17*len(input_var),8*output_shape,2*output_shape,output_shape]
locs = {'Emmaton':0,'Jersey Point':1,'Collinsville':2,'Rock Slough':3}
abbrev_map = {'rock slough':'ORRSL','rockslough':'ORRSL',
            'emmaton':'EMM','jersey point':'JP','jerseypoint':'JP',
            'antioch':'antioch','collinsville':'CO',
            'mallard':'Mallard','mallard island':'Mallard',
            'los vaqueros':'LosVaqueros','losvaqueros':'LosVaqueros',
            'martinez':'MTZ',
            'middle river':'MidR_intake','MiddleRiver':'MidR_intake',
            'victoria cannal':'Victoria_intake','Vict Intake':'Victoria_intake',
            'cvp intake':'CVP_intake','clfct forebay':'CCFB',
            'clfct forebay intake':'CCFB_intake','x2':'X2'};


output_stations = sorted(output_stations,key=lambda x: locs[x])


x_data,y_data = read_data(data_dir,input_var,output_stations)

[x_norm,x_slope,x_bias] = normalize_in(x_data)
[y_norm,y_slope,y_bias] = normalize_in(y_data)

date = np.arange('1940-10',
                 np.datetime64('1940-10') + np.timedelta64(len(x_norm), 'D'),
                 dtype='datetime64[D]')

if not os.path.exists(results_dir):
    os.mkdir(results_dir)
    
for loc in output_stations:
    test_index = np.arange(output_stations.index(loc),output_stations.index(loc)+1)
    ann_name = abbrev_map[loc.lower()]
    tf.reset_default_graph()
    tf.set_random_seed(1)
    x = tf.placeholder(tf.float32, [None, 17*len(input_var)], name='InputData')
    y = tf.placeholder(tf.float32, [None, output_shape], name='LabelData')
    
    # Create some variables.
  
    W1 = tf.get_variable(name='w1',shape=nn_shape[:2],dtype='float32')
    b1 = tf.get_variable(name='b1',shape=nn_shape[1],dtype='float32')
    W2 = tf.get_variable(name='w2',shape=nn_shape[1:3],dtype='float32')
    b2 = tf.get_variable(name='b2',shape=nn_shape[2],dtype='float32')
    W3 = tf.get_variable(name='w3',shape=nn_shape[2:],dtype='float32')
    b3 = tf.get_variable(name='b3',shape=nn_shape[3],dtype='float32')
    
    with tf.name_scope('layer1'):
        first_out = tf.sigmoid(tf.add(tf.matmul(x,W1),b1))
    with tf.name_scope('layer2'):
        second_out = tf.sigmoid(tf.add(tf.matmul(first_out,W2),b2))
    # with tf.name_scope('layer3'):
    #     third_out = tf.sigmoid(tf.add(tf.matmul(second_out,W3),b3))
    with tf.name_scope('layer3'):
        pred = tf.matmul(second_out, W3) + b3
        
        
    feed_dict = {x: x_norm,y: y_norm[:,output_stations.index(loc)].reshape(-1,output_shape)}
    
    saver = tf.train.Saver({'W1':W1,'b1':b1,'W2':W2,'b2':b2,'W3':W3,'b3':b3})
    
    sess = tf.Session()
    print('Testing ANN for %s...' %(output_stations[0]))
    with sess.as_default():
        with sess.graph.as_default():
            saver.restore(sess, os.path.join(model_dir,ann_name,"model.ckpt"))
            print("Model restored, testing...")
            
            y_predicted = sess.run(pred,feed_dict)
            MSE = np.mean(((y_predicted-y_norm[:,test_index])/y_slope[test_index])**2,axis=0)
            MAPE = np.mean(np.abs(y_predicted-y_norm[:,test_index])/y_norm[:,test_index],axis=0)
            print("MSE = %d" % MSE)
            print("MAPE = %.2f%%" % (MAPE*100))
            results = pd.DataFrame(data=((y_predicted-y_bias[test_index])/y_slope[test_index]).reshape(-1,),    # values
                                   index=date,    # 1st column as index
                                   columns=[loc])  # 1st row as the column names
            results.index.name='date    '
            results.to_csv(os.path.join(results_dir,'%s_ANN_results.txt'%ann_name),
                           sep='\t',
                           float_format='%5.4f',
                           header=True,
                           index=True)

            real_data = pd.DataFrame(data=(y_data[:,test_index]).reshape(-1,),    # values
                                   index=date,    # 1st column as index
                                   columns=[loc])  # 1st row as the column names
            real_data.index.name='date    '
            real_data.to_csv(os.path.join(results_dir,'%s_real_data.txt'%ann_name),
                             sep='\t',
                             float_format='%5.4f',
                             header=True,
                             index=True)
            writeF90('fnet_'+abbrev_map[output_stations[0].lower()]+".f90",
                      y_slope[test_index],y_bias[test_index],
                      sess.run(W1).transpose(),sess.run(b1),
                      sess.run(W2).transpose(),sess.run(b2),
                      sess.run(W3).transpose(),sess.run(b3))