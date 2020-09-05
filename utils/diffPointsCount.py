'''
function: counting for difference between label and prediction data
'''
from tqdm import tqdm 
import os
import pandas as pd
import numpy as np

def countDiffPoint(pre_file, label_file, save_file):
    
    # help function: counts of points per line-trace
    def countsPerFile(df_num, LT_list, TVs):
        count = 0 
        last_one_l, last_one_t = df_num[0][0], df_num[0][1]
        for row in tqdm(df_num):
            line = row[0]
            trace = row[1]
            if last_one_l != line or last_one_t != trace:
                LT_list.append(str(last_one_l)+'_'+str(last_one_t))
                TVs.append(count)
                count = 0 
                last_one_l = line
                last_one_t = trace
            count += 1    
        LT_list.append(str(last_one_l)+'_'+str(last_one_t))
        TVs.append(count)
 
    df_pre = pd.read_csv(pre_file, skiprows=0, dtype=int)
    df_label = pd.read_csv(label_file, skiprows=0, dtype=int)
    
    pre_LT = [] # buffer for line-trace 
    pre_TV = [] # buffering TV counts per line-trace 
    pre_num = df_pre.to_numpy()
    countsPerFile(pre_num, pre_LT, pre_TV)
     
    label_LT = []
    label_TV = []
    label_num = df_label.to_numpy() 
    countsPerFile(label_num, label_LT, label_TV)
    
    # counting for points difference per line-trace
    diff_cnt = []
    diff_all = 0
    # TODO: Addressing mismatching between label and prediction
    #for i, item in tqdm(enumerate(pre_LT)):
    #    if label_LT[i] != item:
    #        print("ERROR: predicted do not match with label, which predicted one is %s and label one is %s" %(item, label_LT[i])) 
    #        exit(-1)
    #    tmp_diff = pre_TV[i] - label_TV[i]
    #    diff_cnt.append(tmp_diff)  
    #    diff_all += tmp_diff
    
    # FIXED: mismatching addressed by throw these points which label data lacked of 

    i = 0
    try:
        while label_LT[i]:
            if  label_LT[i] != pre_LT[i]:
                tmp = pre_LT.pop(i)
                pre_TV.pop(i)
                print("ERROR: predicted do not match with label, which predicted one is %s and label one is %s" %(tmp, label_LT[i])) 
                continue 
            print("INFO: i: ", i)
            print(pre_LT[i], label_LT[i])
            tmp_diff = pre_TV[i] - label_TV[i]
            diff_cnt.append(tmp_diff)  
            diff_all += tmp_diff
            i += 1
    except IndexError as err:
        print("INFO DONE, last item line-trace is: {}".format(pre_LT[i-1]))
        
    # compute the mean of different points by hand
    mean_diff = diff_all //  len(diff_cnt)

    # compute mean, median of data
    pre_TV_np = np.array(pre_TV)
    label_TV_np = np.array(label_TV)
    diff_cnt_np = np.array(diff_cnt)

    # format the results  

    # ALL
    pre_LT.append('DiffAll')
    pre_TV.append('DiffAll')
    label_TV.append('DiffAll')
    diff_cnt.append(diff_all)

    # mean
    pre_LT.append('Mean')
    pre_TV.append(int(np.mean(pre_TV_np)))
    label_TV.append(int(np.mean(label_TV_np)))
    diff_cnt.append(int(np.mean(diff_cnt_np)))

    # median
    pre_LT.append('Median')
    pre_TV.append(int(np.median(pre_TV_np)))
    label_TV.append(int(np.median(label_TV_np)))
    diff_cnt.append(int(np.median(diff_cnt_np)))

    
    ## TOFIX: when set dtype=int, there will throw error
    dataframe = pd.DataFrame({'Line-Trace':pre_LT, 'Pred-TVs':pre_TV, 'Lable-TVs':label_TV, 'Dffirence':diff_cnt}, dtype = int)
    dataframe.to_csv(save_file, index=False, sep=',') 
   
    
if __name__ == '__main__':
    pre_file = './dataTest/complex_train_predicted_data.csv' 
    #pre_file = './dataTest/complex_val_predicted_data.csv' 
    #pre_file = './dataTest/complex_test_predicted_data.csv'
    
    label_file = './dataTest/complex_train_label.csv' 
    #label_file = './dataTest/complex_val_label.csv'
    #label_file = './dataTest/complex_test_label.csv'
    
    save_file = './dataTest/complex_train_diff.csv'
    #save_file = './dataTest/complex_val_diff.csv'
    #save_file = './dataTest/complex_test_diff.csv'

    #Testing code
    #df = pd.read_csv(pre_file, skiprows=0, dtype=int)
    #count = 0
    #df_num = df.to_numpy()
    #rows, cols = df_num.shape
    #for row in df_num:
    #    if count>10:
    #        break
    #    print(row[1])
    #    #for row in df.ix[i]:
    #    count += 1

    if not os.path.exists(save_file):
        with open(save_file, 'w+') as f:
            pass
    countDiffPoint(pre_file, label_file, save_file) 
