'''
function: format csv file from label data and train, val, test images

'''


import os 
import pandas as pd
from tqdm import tqdm
from readFromDat import getSpecCMPData

def getRespectLable(imgpath, label_file, whole_data):
    imglist= os.listdir(imgpath)
    length=len(imglist)
    for i in tqdm(range(length)):
        if not imglist[i].endswith('.png'):
            continue
        str = imglist[i].split('_')
        # need to modify with file name format changing
        linenum = int(str[5])
        cmpnum = int(str[6][:-4])
        data = getSpecCMPData(linenum, cmpnum, label_file)
        if len(data) == 0:  # addressing label miss problem
            continue
        whole_data += data

# using pd function to write the TV data respecte with lin and trace number

def saveToCsv(csv_file):
    for row in tqdm(whole_data):
        items = row.split(',')
        line.append(int(items[0]))
        trace.append(int(items[1]))
        time.append(int(items[2]))
        velocity.append(int(items[3]))

    # TOFIX: when trace above 800 and below 1000, the format is wrong. data type should be set as int but there will eror throwing 
    #dataframe = pd.DataFrame({'line':line, 'trace':trace, 'time':time, 'velocity':velocity}, dtype=int)
    dataframe = pd.DataFrame({'line':line, 'trace':trace, 'time':time, 'velocity':velocity}, dtype=int)
    #dataframe.to_csv(csv_file2, index=False, sep=',')
    dataframe = dataframe.sort_values(by=['line','trace'], ascending=True)
    dataframe.to_csv(csv_file, index=False, sep=',')

if __name__ == '__main__':

    #file_name = '/Users/middle_prince/Desktop/tempt/Lab/Projects/Velocity-Time/data-and-more/training-dataset/datasets/coco-new-complex/train'
    #file_name = '/Users/middle_prince/Desktop/tempt/Lab/Projects/Velocity-Time/data-and-more/training-dataset/datasets/coco-new-complex/val'
    file_name = '/Users/middle_prince/Desktop/tempt/Lab/Projects/Velocity-Time/data-and-more/training-dataset/datasets/coco-new-complex/test'
    
    lable_path = './dataTest/complex-label2.txt'
    
    #csv_file = './dataTest/complex_train_label2.csv'
    #csv_file = './dataTest/complex_val_label2.csv'
    csv_file = './dataTest/complex_test_label2.csv'
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w+') as f:
            pass

    whole_data = [] 
    line = []
    trace = []
    time = []
    velocity = []

    getRespectLable(file_name, lable_path, whole_data)
    saveToCsv(csv_file)

    


