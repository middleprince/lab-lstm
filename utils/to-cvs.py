# transfer vlocity text to cvs file

import io
import numpy as np
import pandas as pd
import csv
    
def transfer(from_path, to_path):
    reader=io.open(from_path, encoding='utf_8_sig')
    list_data=reader.readlines()
    columns=list_data[0].split()
    list=[]
    for i in list_data[1:]:
        list.append(i.split())
    with io.open("to_path","wb") as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(list)
    df=pd.read_csv("to_path",error_bad_lines=False)
    df=pd.DataFrame(df)
    df=df.dropna(axis=0,how='all',inplace=False)#当所有的行为NaN时，删除该行
    result=df[['desc','recom_code','num','vvctr','time_ctr']]#切片
    result=pd.pivot_table(result,index=['num'],columns=['desc','recom_code'])#拉透视表，透视表对数据格式有要求
    result.to_csv("to_path",sep=',',header=True,encoding='utf_8_sig')#存储结果
    
def transfer_simple(from_path, to_path):
    data_txt = np.loadtxt(from_path)
    data_txtDF = pd.DataFrame(data_txt)
    data_txtDF.to_csv(to_path,index=False)

if __name__ == '__main__':
    from_path = "../data/data.txt"
    to_path = "../data/velocity-time-large.csv"
    transfer_simple(from_path, to_path)

