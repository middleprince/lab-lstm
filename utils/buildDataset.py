'''
function: build training dataset which format:{line,trace, time, velocity, delta time, delta velocity}
'''
from tqdm import tqdm 
import os
import pandas as pd
import numpy as np

def  buildDataset(pre_file, label_file, save_file):

    # helper function : create line-trace dicitonary for manipulation latter
    # the data format is {'line_tv':[(t, v), (t, v), ..., (t, v)], ..., }
    def creatDic(df_np, LT_dic):
        tvs = [ ]
        last_one_l, last_one_t = df_np[0][0], df_np[0][1]
        for row in tqdm(df_np):
            line = row[0]
            trace = row[1]
            time = row[2]
            velocity = row[3]

            if last_one_l != line or last_one_t != trace:
                LT_dic[str(last_one_l)+'_'+str(last_one_t)] = tvs
                last_one_l = line
                last_one_t = trace
                tvs = [ ]
            tvs.append((time, velocity))    
        LT_dic[str(last_one_l)+'_'+str(last_one_t)] = tvs

    df_pre = pd.read_csv(pre_file, skiprows=0, dtype=int)
    df_label = pd.read_csv(label_file, skiprows=0, dtype=int)

    pre_LTDic = {} # buffering for tvs with respective with line-trace
    pre_np = df_pre.to_numpy()
    creatDic(pre_np, pre_LTDic)
    
    label_LTDic = {}
    label_np = df_label.to_numpy()
    creatDic(label_np, label_LTDic)

    # buffer for delta-tv to storing
    delta_time = []
    delta_velocity = []

    # buffer for line and trace and time, velocity
    line_list = [ ]
    trace_list = [ ]
    pre_time = [ ]
    pre_velocity = [ ]

    # compute delta tv per line-trace
    for line_trace, tvs in tqdm(zip(label_LTDic.keys(), label_LTDic.values())):
        # TODO: FIX match error 
        if not pre_LTDic[line_trace]:
            print("ERROR: the {} is not exit in prediction file".format(line_trace))
            exit(-1)
        tvs_pre = pre_LTDic[line_trace]

        lt_str = line_trace.split('_')
        line_tmp = int(lt_str[0])
        trace_tmp = int(lt_str[1])

        # with respect to prediction tvs
        i = 0
        j = 0
        tvs_num = len(tvs_pre)
        while i < tvs_num:
        #for i in range(len(tvs_pre)):
            
            line_list.append(line_tmp)
            trace_list.append(trace_tmp)
            pre_time.append(tvs_pre[i][0])
            pre_velocity.append(tvs_pre[i][1])

            # TODO: there is no predicaton when time less than 400 in complicated distriction.
            # the data set should remove these data in labels
            try:
                while tvs[j][0] < 400:
                    j += 1
                    #print("INFO the label less than 400 index is {}:".format(j))
            except IndexError as error:
                pass 
                #print(error)
                #print("INFO line trace is {}".format(lt_str))
                #print("INFO the label index is {}:".format(j))
                #print("INFO the prediction index is {}:".format(i))
                #exit(-1)
            
            # FIXED: handling index exception
            try:
                delta_t = tvs[j][0] - tvs_pre[i][0]
                delta_v = tvs[j][1] - tvs_pre[i][1]
                delta_time.append(delta_t) 
                delta_velocity.append(delta_v)
            except IndexError as error:
                delta_time.append(tvs[-1][0] - tvs_pre[i][0])    
                delta_velocity.append(tvs[-1][1] - tvs_pre[i][0])    
            i += 1
            j += 1



    df = pd.DataFrame({'line':line_list, 'trace':trace_list, 'time':pre_time, 'velocity':pre_velocity, 'delta time':delta_time, 'delta velocity':delta_velocity}, dtype=int)
    df.to_csv(save_file, index=False, sep=',')

if __name__ == '__main__':
    
    ## testing for dic iterator code
    #demo = {'one':[(1,2), (4, 9)], 'three':[(2,4), (99, 100)]}
    #for key, value in tqdm(zip(demo.keys(), demo.values())):
    #    for i in range(len(demo)):
    #        print(value[i][0])

    
    #pre_file = './dataTest/complex_train_withFCOS.csv' 
    #pre_file = './dataTest/complex_val_withFCOS.csv' 
    #pre_file = './dataTest/complex_test_withFCOS.csv'
    #pre_file = './dataTest/data_train_hade_FCOS.csv'
    pre_file = './dataTest/data_val_hade_FCOS.csv'
    
    #label_file = './dataTest/complex_train_label2.csv' 
    #label_file = './dataTest/complex_val_label2.csv'
    #label_file = './dataTest/complex_test_label2.csv'
    #label_file = './dataTest/data_train_hade_label.csv'
    label_file = './dataTest/data_val_hade_label.csv'
    

    #save_file = './dataTest/complex_train_lstm_fcos.csv'
    #save_file = './dataTest/complex_val_lstm_fcos.csv'
    #save_file = './dataTest/complex_test_lstm_fcos.csv'
    #save_file = './dataTest/data_train_lstm_hade.csv'
    save_file = './dataTest/data_val_lstm_hade.csv'

   
    if not os.path.exists(save_file):
        with open(save_file, 'w+') as f:
            pass
    
    buildDataset(pre_file, label_file, save_file)
