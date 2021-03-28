#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:59:10 2019

@author: siyuqi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
def read_data(xl_path,input_var,output_stations):
    xls = pd.ExcelFile(xl_path)
    WS_in = pd.read_excel(xls,0,header=None)
    # drop first column if it's date
    if type(WS_in.iloc[0][0]) != str and type(WS_in.iloc[0][0]) !=float:
        WS_in.drop(columns=[0],inplace=True)
    # drop empty rows and columns
    WS_in.dropna(how='all',inplace=True)
    WS_in.dropna(1,how='all',inplace=True)
    WS_in.columns = range(len(WS_in.columns))
    WS_in.reset_index(drop=True, inplace=True)
    
    # select input parameters
    drop_idx = [i for i in WS_in.iloc[0] if type(i)==str]
    drop_idx = Filter(drop_idx,input_var)
    WS_in.drop(columns=drop_idx,inplace=True)
    # drop headers
    WS_in=WS_in[WS_in.applymap(np.isreal).all(1)]
    WS_in.reset_index(drop=True, inplace=True)
    
    rows_with_nan = WS_in.isna().any(axis=1)
    WS_in.drop([ii for ii in range(len(rows_with_nan))
                        if rows_with_nan.at[ii]],inplace=True)
    WS_in.reset_index(drop=True, inplace=True)
    
    # read output data
    WS_out = pd.read_excel(xls,1,header=None)
    # drop first column if it's date
    if type(WS_out.iloc[0][0]) != str and type(WS_out.iloc[0][0]) !=float:
        WS_out.drop(columns=[0],inplace=True)
    WS_out.columns = range(len(WS_out.columns))


    # select output station
    drop_idx = [i for i in WS_out.iloc[0] if type(i)==str]
    drop_idx = Filter(drop_idx,output_stations)

    WS_out.drop(columns=drop_idx,inplace=True)

    # drop empty rows and columns
    WS_out.dropna(how='all',inplace=True)
    WS_out.dropna(1,how='all',inplace=True)
    WS_out.reset_index(drop=True, inplace=True)
    # drop headers
    WS_out=WS_out[WS_out.applymap(np.isreal).all(1)]
    WS_out.reset_index(drop=True, inplace=True)
    # drop data with missing input features
    WS_out.drop([ii for ii in range(len(rows_with_nan))
                    if rows_with_nan.at[ii]],inplace=True)
    
    x_data = np.array(WS_in).astype('float32')
    y_data = np.array(WS_out).astype('float32')

    if len(x_data) > len(y_data):
        print('Disgarding last %d row(s) of data in output set...'
                  %(len(x_data)-len(y_data)))
        x_data=np.delete(x_data,
                          np.arange(len(y_data),len(x_data)),
                          0)
    elif len(x_data) < len(y_data):
        print('Disgarding last %d row(s) of data in input set...'
              %(len(y_data)-len(x_data)))
        y_data=np.delete(y_data,
                          np.arange(len(x_data),len(y_data)),
                          0)

    return [x_data,y_data]
    
def writeF90(f90path,name,slope,bias,w1=None,b1=None,w2=None,b2=None,w3=None,b3=None):
    with open(os.path.join(f90path,'fnet_'+name+".f90"),"w") as f:
        f.write('module fnet_'+name+'\n')
        if type(slope)==list or type(slope)==np.ndarray:
            f.write('\n! a = ')
            np.savetxt(f,slope.reshape(1,-1),fmt='%.8f', delimiter=', ',newline='')
            f.write('\n! b = ')
            np.savetxt(f,bias.reshape(1,-1),fmt='%.8f', delimiter=', ',newline='')
            f.write('\n\n' )
        else:
            f.write('\n! a = %.8f\n! b = %.5f\n\n' %(slope,bias))

        f.write('\nintrinsic Reshape\n')
        f.write('real, dimension(%d, %d) :: input = &\n' %(w1.shape[0],w1.shape[1]))
        f.write('  Reshape((/')
        save12(f,w1)
        f.write('/),(/%d,%d/))\n'%(w1.shape[0],w1.shape[1]))
        ii=1
        for ww in [w2,w3]:
            f.write('real, dimension(%d,%d) :: hidden%d = &\n' %(ww.shape[0],ww.shape[1],ii))
            f.write('  Reshape((/')
            save12(f,ww)
            f.write('/),(/%d,%d/))\n'%(ww.shape[0],ww.shape[1]))
            ii+=1

        ii=1
        for bb in [b1,b2,b3]:
            f.write('real, dimension(%d) :: bias%d = &\n' %(bb.shape[0],ii))
            f.write('  (/')
            save12(f,bb)
            f.write('/)\n')
            ii+=1
        f.write('contains\n')
        f.write('subroutine fnet_'+name+'_initall()\n')
        f.write('end subroutine fnet_'+name+'_initall\n')
        f.write('subroutine fnet_'+name+'_engine(inarray, outarray, init)\n')
        f.write('  intrinsic MatMul, Size\n')
        f.write('  real, dimension(:), intent(in) :: inarray\n')
        f.write('  real, dimension(:), intent(inout) :: outarray\n')
        
        f.write('  real, dimension(%d) :: inarray2\n' % w1.shape[1])
        
        f.write('  real (kind=8), dimension(%d) :: layer1\n' % w1.shape[0])
        f.write('  real (kind=8), dimension(%d) :: layer2\n' % w2.shape[0])
        f.write('  real (kind=8), dimension(%d) :: layer3\n' % w3.shape[0])
        f.write('  integer , intent(inout) :: init\n')
        f.write('  integer :: i, j\n')
        f.write('  do i = 1, %d\n' % w1.shape[1])
    
        f.write('    inarray2(i) = inarray(%d-i)\n' % (w1.shape[1]+1))
        f.write('  end do\n')
        f.write('  layer1 = MatMul(input,inarray2)\n')
        f.write('  layer1 = layer1 + bias1\n')
        f.write('  do i = 1, Size(layer1,1)\n')
        f.write('    layer1(i) = 1.0 / (1.0 + DEXP(-1.0 * layer1(i)))\n')
        f.write('  end do\n')
        f.write('  layer2 = MatMul(hidden1,layer1)\n')
        f.write('  layer2 = layer2 + bias2\n')
        f.write('  do i = 1, Size(layer2,1)\n')
        f.write('    layer2(i) = 1.0 / (1.0 + DEXP(-1.0 * layer2(i)))\n')
        f.write('  end do\n')
        f.write('  layer3 = MatMul(hidden2,layer2)\n')
        f.write('  layer3 = layer3 + bias3\n')
        f.write('  outarray(1) = layer3(1)\n')
        f.write('end subroutine fnet_'+name+'_engine\n')
        f.write('end module fnet_'+name+'\n')
    return

def save12(f,array):
    if len(array.shape)==2:
        line_indent='            '
    else:
        line_indent='  '
    if array.size % 12 == 0:
        subarray = array.flatten()[:-12]
        np.savetxt(f, subarray.reshape(-1,12), fmt='%.8f', delimiter=',',newline=' &\n'+line_indent+',') 
        subarray = array.flatten()[-12:]
        np.savetxt(f, subarray.reshape(1,-1), fmt='%.8f', delimiter=',',newline=' &\n'+line_indent+',') 
    else:
        subarray = array.flatten()[:int(np.ceil(array.size/12-1)*12)]
        np.savetxt(f, subarray.reshape(-1,12), fmt='%.8f', delimiter=',',newline=' &\n'+line_indent+',') 
        subarray = array.flatten()[int(np.ceil(array.size/12-1)*12):]
        np.savetxt(f, subarray.reshape(1,-1), fmt='%.8f', delimiter=',',newline='') 

def normalize_in(ori_data,hi=0.9,lo=0.1):
    data_max = np.max(ori_data,axis=0)
    data_min = np.min(ori_data,axis=0)
    
    norm_slope = np.divide((hi-lo), 
                           (data_max-data_min), 
                           out=np.zeros_like(data_max),
                           where=(data_max-data_min)!=0)
    norm_bias = lo - norm_slope * data_min
    return [norm_slope * ori_data + norm_bias,norm_slope,norm_bias]

def initnw(in_out_num_list,x_in):
    """
    Nguyen-Widrow initialization function
    
    :Parameters:
        layer: core.Layer object
            Initialization layer
    """
    all_w = []
    all_b = []
    for in_out_num in in_out_num_list:
        ci = in_out_num[0]
        cn = in_out_num[1]
        w_fix = 0.7 * cn ** (1. / ci)
        w_rand = np.random.rand(cn, ci) * 2 - 1
        # Normalize
        if ci == 1:
            w_rand = w_rand / np.abs(w_rand)
        else:
            w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(cn, 1)) * w_rand

        ww = w_fix * w_rand
        bb = np.array([0]) if cn == 1 else w_fix * np.linspace(-1, 1, cn) * np.sign(ww[:, 0])

        amin = -2
        amax = 2

        x = 0.5 * (amax - amin)
        y = 0.5 * (amax + amin)
        ww = x * ww
        bb = x * bb + y

        # Scaleble to inp_minmax
        minmax = np.stack([np.min(x_in,axis=0), np.max(x_in,axis=0)],axis=1)
        minmax[np.isneginf(minmax)] = -1
        minmax[np.isinf(minmax)] = 1

        x = np.divide(np.asarray([2.]),(minmax[:, 1] - minmax[:, 0]),\
                      out=np.zeros([minmax.shape[0],]),\
                      where=(minmax[:, 1] - minmax[:, 0])!=0)
        y = 1. - minmax[:, 1] * x
        ww = ww * x
        bb = np.dot(ww, y) + bb
        all_w.append(ww.T)
        all_b.append(bb)
        x_in = np.matmul(x_in,ww.T) + bb
    return list(zip(all_w,all_b))


def Filter(string, substr): 
    return [ii for ii in range(len(string))
        if not any(sub.lower() in string[ii].lower() for sub in substr)]        

def show_eval(y_train_predicted,y_train0,y_test_predicted,y_test0,y_slope,y_bias,name=None):
    """inputs are vectors"""
    err = np.divide(y_train_predicted-y_train0,y_train0,\
                   out=np.zeros_like(y_train0), where=(y_train0!=0))
    print('train mape:  %.4f'% np.mean(np.abs(err),axis=0))
    print('train mse:  %.2f' % np.mean(((y_train_predicted-y_train0)/y_slope)**2,axis=0))
    err = np.divide(y_test_predicted-y_test0,y_test0,\
                   out=np.zeros_like(y_test0), where=(y_test0!=0))
    print('test mape:  %.4f'% np.mean(np.abs(err),axis=0))
    print('test mse: %.2f'% np.mean(((y_test_predicted-y_test0)/y_slope)**2,axis=0))
    if name:
        # linear curve fitting
        fig = plt.figure()
        slope, intercept, r_value, p_value, std_err = \
                        stats.linregress(y_test0,y_test_predicted)
        line = slope*y_test0+intercept
        plt.plot((y_test0-y_bias)/y_slope,
                 (y_test_predicted-y_bias)/y_slope,
                 linestyle="",marker="o",markersize = 2)
        plt.plot((y_test0-y_bias)/y_slope,
                 (line-y_bias)/y_slope,'k')
        plt.xlabel('Actual Target')
        plt.ylabel('Predicted')
        fig.text(0.2, 0.7, 'y=%.4fx\n $R^2=$%.4f'%(slope,r_value), fontsize=14)
        fig.savefig('%s.jpg'%(name), format='jpg', dpi=100)
        
   
def read_csv(input_data_path,output_data_path,input_var_list,output_var_list):
    inputs = pd.read_csv(input_data_path,header=0,index_col=0,comment='#')
    # select input parameters
    drop_idx = list(inputs.columns.values)
    drop_idx = Filter(drop_idx,input_var_list)
    inputs.drop(columns=drop_idx,inplace=True)
    # sort input data by input_var_list
    inputs.reindex(columns=input_var_list)
    inputs=inputs.fillna(0)
    
    outputs = pd.read_csv(output_data_path,header=0,index_col=0,comment='#')
    # select output station
    drop_idx = list(outputs.columns.values)
    drop_idx = Filter(drop_idx,output_var_list)
    outputs.drop(columns=drop_idx,inplace=True)
    # sort outputs data by output_var_list
    outputs.reindex(columns=output_var_list)
            
    return [np.array(inputs).astype('float32'),
            np.array(outputs).astype('float32')]

def read_csv_mix(input_daily_path,daily_var_list,input_monthly_path,monthly_var_list,
                 output_data_path, output_var_list):
    ### input_daily_path: the new dataset
    ### input_monthly_path: the old dataset
    inputs_d = pd.read_csv(input_daily_path,header=0,index_col=0,comment='#')
    inputs_d.index = pd.to_datetime(inputs_d.index)
    # select input parameters
    drop_idx = list(inputs_d.columns.values)
    drop_idx = Filter(drop_idx,daily_var_list)
    inputs_d.drop(columns=drop_idx,inplace=True)
    # sort input data by input_var_list
    inputs_d.reindex(columns=daily_var_list)
    inputs_d=inputs_d.fillna(0)
    
    # read old dataset and retrieve first day of the month for each date
    inputs_m = pd.read_csv(input_monthly_path,header=0,index_col=0,comment='#')
    inputs_m.index = pd.to_datetime(inputs_m.index)
    # select input parameters
    drop_idx = list(inputs_m.columns.values)
    drop_idx = Filter(drop_idx,monthly_var_list)
    inputs_m.drop(columns=drop_idx,inplace=True)
    # sort input data by monthly_var_list
    inputs_m.reindex(columns=monthly_var_list)
    inputs_m=inputs_m.fillna(0)
    
    inputs_m.reset_index(inplace=True)
    inputs_m['month'] = inputs_m['index'].apply(lambda x : x.replace(day=1))

    ## add a column representing month index to inputs_d
    inputs_d['month']= inputs_d.index.to_series().apply(lambda x : (x.to_period('m')- inputs_m['index'][0].to_period('m')).n)
    
    ## delete duplicate rows in inputs_m
    inputs_m.set_index('month', drop=True, inplace=True)
    inputs_m.drop(columns='index',inplace=True)
    inputs_m = inputs_m[~inputs_m.index.duplicated(keep='first')]

    # inputs_m['index'].apply(lambda x : x+pd.DateOffset(months=2))

    outputs = pd.read_csv(output_data_path,header=0,index_col=0,comment='#')
    # select output station
    drop_idx = list(outputs.columns.values)
    drop_idx = Filter(drop_idx,output_var_list)
    outputs.drop(columns=drop_idx,inplace=True)
    # sort outputs data by output_var_list
    outputs.reindex(columns=output_var_list)
            
    return [np.array(inputs_d).astype('float32'),
            np.array(inputs_m).astype('float32'),
            np.array(outputs).astype('float32')]


def process_data(input_dataset,output_dataset,
                 single_days=7, window_num=10,
                 window_size = 11,predict=0,
                 use_input = True,add_ec=False):
    assert len(input_dataset)==len(output_dataset), "input and output data must have same length"
    data = []
    labels = []
    total_avg_num = window_num*window_size
    end_date = len(input_dataset)-predict
    if window_num != 0 and window_size != 0:
        for i in range(single_days+total_avg_num+int(add_ec)-1, end_date):
            # Reshape data from (history_size,# of variables) to
            # new_shape
            if use_input:
                input_data = np.concatenate((np.reshape(input_dataset[i:i-single_days:-1], (single_days,-1)),
                                                np.mean(np.reshape(input_dataset[i-single_days:(None if i-single_days-total_avg_num==-1 else i-single_days-total_avg_num):-1], (window_num,window_size,-1)),axis=1)),axis=0)
            if add_ec:
                ec_data = np.concatenate([np.reshape(output_dataset[i-1:i-single_days-1:-1], (single_days,-1)),
                                          np.mean(np.reshape(output_dataset[i-single_days-1:(None if i-single_days-total_avg_num-1==-1 else i-single_days-total_avg_num-1):-1], (window_num,window_size,-1)),axis=1)],axis=0)
            
            if add_ec and use_input:
                data.append(np.concatenate([input_data,ec_data],axis=1))
            elif add_ec:
                data.append(ec_data)
            elif use_input:
                data.append(input_data)
            else:
                print("No input data selected")
                return
            labels.append(output_dataset[i+predict])
    else:
        for i in range(single_days+total_avg_num+int(add_ec)-1, end_date):
            # Reshape data from (history_size,# of variables) to
            # new_shape
            if use_input:
                input_data = np.reshape(input_dataset[i:(None if i-single_days==-1 else i-single_days):-1], (single_days,-1))
            if add_ec:
                ec_data = np.reshape(output_dataset[i-1:(None if i-single_days-1==-1 else i-single_days-1):-1], (single_days,-1))
                
            if add_ec and use_input:
                data.append(np.concatenate([input_data,ec_data],axis=1))
            elif add_ec:
                data.append(ec_data)
            elif use_input:
                data.append(input_data)
            else:
                print("No input data selected")
                return
            labels.append(output_dataset[i+predict])
            
    return np.array(data).transpose(0,2,1), np.array(labels)

def process_data_vary_pred(input_dataset,output_dataset,
                           single_days=7, window_num=10,
                           window_size = 11,predict_list=None):
    assert len(input_dataset)==len(output_dataset), "input and output data must have same length"
    assert len(predict_list)==output_dataset.shape[1]
    assert min(predict_list) >= 0
    data = []
    labels = []
    total_avg_num = window_num*window_size
    end_date = len(input_dataset)-max(predict_list)
    if window_num != 0 and window_size != 0:
        for i in range(single_days+total_avg_num-1, end_date):
            # Reshape data from (history_size,# of variables) to
            # new_shape
            data.append(np.concatenate((np.reshape(input_dataset[i:i-single_days:-1], (single_days,-1)),
                                        np.mean(np.reshape(input_dataset[i-single_days:i-single_days-total_avg_num:-1], (window_num,window_size,-1)),axis=1)),axis=0))
            labels.append(np.asarray([output_dataset[i+predict,ii] for ii,predict in enumerate(predict_list)]))
    else:
        for i in range(single_days+total_avg_num-1, end_date):
            # Reshape data from (history_size,# of variables) to
            # new_shape
            if i-single_days==-1:
                data.append(np.reshape(input_dataset[i::-1], (single_days,-1)))
            else:
                data.append(np.reshape(input_dataset[i:i-single_days:-1], (single_days,-1)))
            labels.append(np.asarray([output_dataset[i+predict,ii] for ii,predict in enumerate(predict_list)]))
            
    return np.array(data).transpose(0,2,1), np.array(labels)

  
  
def conv_filter_generator(single_days=7,window_num=10,window_size = 11):
    w = np.zeros((1,single_days+window_num*window_size,single_days+window_num))
    for ii in range(single_days):
        w[0,single_days+window_num*window_size-ii-1,single_days-ii-1] = 1
    for ii in range(window_num):
        w[0,((window_num-ii-1)*window_size):((window_num-ii)*window_size),single_days+ii] = 1/window_size
    return w
  
def conv_filter_generator_v2(single_days=7,window_num=10,window_size = 11,var_num=7):
    w1 = np.repeat(np.identity(var_num)[np.newaxis,...],1, axis=0)
    w2 = np.repeat(np.identity(var_num)[np.newaxis,...]/window_size,window_size, axis=0)
    return (w1,w2)
  
def add_noise(data, noise_dict):
    ## add noise to data according to noise_dict
    ## data must have the shape of (no. of samples, no. of variables)
    noisy_data = np.copy(data)
    p_signal = np.mean(data**2,axis=0)
    for index in noise_dict.keys():
        p_noise = p_signal[index] / (10**(noise_dict[index]/10))
        noise = np.random.normal(0, np.sqrt(p_noise), data.shape[0])
        noisy_data[:,index] = noise+data[:,index]
    return noisy_data