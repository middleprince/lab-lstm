# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model.model_pytorch import train, predict

class Config:
    # 数据参数
    feature_columns = [0, 1]     # 要作为feature的列,在数据处理时去除了line与trace则t,v 从0开始;也就是(t, v), 數據格式(line, trace, time, velocity, delta_t, delta_v)
    label_columns = [2, 3]                  # 要预测的列，也即是label data:(delta_t delta_v)
    
    # TOFIX: 當前的情況feature與label是不存在重疊的
    #label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)  # feature 與 label的數據存在重疊的情況

    predict_length = 20             # 需要預測的序列數量

    # 网络参数
    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 128           # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2             # LSTM的堆叠层数
    dropout_rate = 0.2          # dropout概率
    time_step = 26              # LSTM的time step 序列长度,平均的GT中有26個點

    

    # 训练参数
    do_train = True
    do_predict = False
    add_train = False           # 是否载入已有模型参数进行增量训练
    shuffle_train_data = False   # 是否对训练数据做shuffle
    use_cuda = True            # 是否使用GPU训练

    # DONE：更改train、val、test的來源，直接從外部讀取
    #train_data_rate = 0.90      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    #valid_data_rate = 0.15      # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 32
    learning_rate = 0.001
    epoch = 150                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 70                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42            # 随机种子，保证可复现

    do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state,使用道间相似性 
    continue_flag = ""           # TODO.但实际效果不佳，可能原因：仅能以 batch_size = 1 训练
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # Debug Mode
    debug_mode = False 
    debug_num = 10000  # 仅用debug_num条数据来调试

    # 框架参数
    model_name = "model_" + continue_flag 

    ## 路径参数
     
    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    train_data_path = "./data/train_lstm.csv"
    val_data_path = "./data/val_lstm.csv"
    test_data_path = "./data/test_lstm.csv"
    prediction_file_path = "./data/test_prediction.csv"

    #model_save_path = "./checkpoint/" + cur_time + '_' + "/"
    model_save_path = "./checkpoint/" + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_save = True                  # 是否将config和训练过程记录到log
    do_figure_save = False
    do_train_visualized = False          # 训练loss可视化，pytorch用visdom
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + "/"
        os.makedirs(log_save_path)


class Data:
    def __init__(self, config):
        self.config = config
        # load feaure and label data 
        #self.data_train_x, self.data_train_x_name = self.read_data('train', 'feature')
        #self.data_val_x, self.data_val_x_name = self.read_data('val', 'feature')
        #self.data_test_x, self.data_test_x_name = self.read_data('test', 'feature')

        #self.data_train_y, self.data_train_y_name = self.read_data('train', 'label')
        #self.data_val_y, self.data_val_y_name = self.read_data('val', 'label')
        #self.data_test_y, self.data_test_y_name = self.read_data('test', 'label')
        #self.data_num = self.data.shape[0]
        
        
        self.data_train, self.data_train_name = self.read_data('train')
        self.data_val, self.data_val_name = self.read_data('val')
        self.data_test, self.data_test_name = self.read_data('test')

        
        self.train_num = self.data_train.shape[0]

        self.train_samples = self.data_train.shape[0] // config.time_step
        self.val_samples = self.data_val.shape[0] // config.time_step

        # DONE: 使用所有的数据进行归一化，从config中取得值 

        self.whole_data = self.data_train
        self.whole_data = np.vstack((self.whole_data, self.data_val))
        self.whole_data = np.vstack((self.whole_data, self.data_test))

        self.mean = np.mean(self.whole_data, axis=0)
        self.std = np.std(self.whole_data, axis=0)
        
        self.norm_data_train = (self.data_train - self.mean) / self.std
        self.norm_data_val = (self.data_val - self.mean) / self.std
        self.norm_data_test = (self.data_test - self.mean) / self.std

        self.start_num_in_test = 0      # 测试集中前t个time-step数据会被删掉，因为它不够一个time_step
    
    # flag声明读取不同的数据集
    #def read_data(self, flag1, flag2):                # 读取初始数据
    #    tmp = flag1+'_data_path' 
    #    data_path = self.config.(eval(tmp))
    #    tmp = flag2+ '_columns' 
    #    data_cols = self.config.(eval(tmp)) 
    #    if self.config.debug_mode:
    #        init_data = pd.read_csv(data_path, nrows=self.config.debug_num,
    #                                usecols=data_cols)
    #    else:
    #        init_data = pd.read_csv(data_path, usecols=data_cols)
    #    return init_data.values, init_data.columns.tolist()     # .columns.tolist() 是获取列名

    # flag声明读取不同的数据集

    def read_data(self, flag):                # 读取初始数据
        if flag == 'train':
            data_path = self.config.train_data_path 
        elif flag == 'val':
            data_path = self.config.val_data_path
        else:
            data_path = self.config.test_data_path
        # 不会读取line与trace
        if self.config.debug_mode:
            init_data = pd.read_csv(data_path, nrows=self.config.debug_num,
                    usecols=[2, 3, 4, 5])
        else:
            init_data = pd.read_csv(data_path, usecols=[2, 3, 4, 5]) # 读取tv与delta-tv 也就是label
        return init_data.values, init_data.columns.tolist()     # .columns.tolist() 是获取列名

    # TODO: 数据获取，确保连续训练下的正确性
    #def create_dataset(self, flag):
    #    feature_data = self.norm_data_train[:,:2]
    #    label_data = self.norm_data_train[self.config.predict_length : self.config.predict_length + self.train_num,
    #            self.config.label_in_feature_index]    # TODO:将input与predicted错位，使用语言模型的设计:label是与input错开predict_length time-sequence此设计可能需要调整

    #    if not self.config.do_continue_train:
    #        # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，eg.：1-20行，2-21行。。。。
    #        train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
    #        train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
    #    else:
    #        # TODO:在连续训练模式下，每time_step行数据会作为一个样本，两个样本错开time_step行，
    #        # 比如：1-20行，21-40行。。。到数据末尾，然后又是 2-21行，22-41行。。。到数据末尾，……
    #        # 这样才可以把上一个样本的final_state作为下一个样本的init_state，而且不能shuffle
    #        train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
    #                   for start_index in range(self.config.time_step)
    #                   for i in range((self.train_num - start_index) // self.config.time_step)]
    #        train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
    #                   for start_index in range(self.config.time_step)
    #                   for i in range((self.train_num - start_index) // self.config.time_step)]

    #    train_x, train_y = np.array(train_x), np.array(train_y)

    #    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
    #                                                          random_state=self.config.random_seed,
    #                                                          shuffle=self.config.shuffle_train_data)   # 划分训练和验证集，并打乱
    #    return train_x, valid_x, train_y, valid_y

    # 建立train、val相应的数据标签集，flag指名当前数据集类型:train, val
    def get_dataset(self, flag):
        if flag == 'train':
            dataset = self.norm_data_train
            samples = self.train_samples
        else:
            dataset = self.norm_data_val
            samples = self.val_samples
        feature_data = dataset[:,self.config.feature_columns]
        ## TODO:不采用语言模型的方法来做label，将t输出直接作为偏移
        label_data = dataset[:, self.config.label_columns]
        
        # 原设计
        # TODO:将input与predicted错位，使用语言模型的设计:label是与input错开predict_length time-sequence此设计可能需要调整
        #label_data = dataset[self.config.predict_length : self.config.predict_length + dataset.shape[0], 
        #        self.config.label_columns]    

        if not self.config.do_continue_train:
            # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，eg.：1-20行，2-21行。。。。
            train_x = [feature_data[i:i+self.config.time_step] for i in range(samples)]
            train_y = [label_data[i:i+self.config.time_step] for i in range(samples)]
        else:
            # TODO:在连续训练模式下，实现两个batch之间的 hidden state可以相互传递; 完成序列的连续性。
            # 目前直接取了序列长度为每个道集下点的个数，暂时不用考虑连续训练的问题.
            # 实现连续训练，当序列长度time_step 小于实际的 tv/cmp 可以在先将数据格式化为(batch_size, batch_len) ,在batch_size的维度上做time_step长度的数据获取。
            # 这样就可以保证在batch_size 维度下被切断的序列时序列首尾连续。

            #TODO:实现index来实现位置的操作之后再完成对实际数据的产生。
            batch_len = samples // self.config.batch_size
            #continue_data = 
            train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]

        
        # FIXED: BUG cant not converte thess numpy object  into pytorch tenseor
        train_x, train_y = np.array(train_x), np.array(train_y)
        
        

        #train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
        #                                                      random_state=self.config.random_seed,
        #                                                      shuffle=self.config.shuffle_train_data)   # 划分训练和验证集，并打乱
        return train_x, train_y

   


    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data_test[:,self.config.feature_columns]
        self.start_num_in_test = feature_data.shape[0] % self.config.time_step  # 这些天的数据不够一个time_step
        samples  = feature_data.shape[0] // self.config.time_step

        # 在测试数据中，每time_step行数据会作为一个样本，两个样本错开time_step行
        # 比如：1-20行，21-40行。。。到数据末尾。
        test_x = [feature_data[i*self.config.time_step : (i+1)*self.config.time_step]
                   for i in range(samples)]
        if return_label_data:       # 实际应用中的测试集是没有label数据的
            label_data = [self.norm_data_test[i*self.config.time_step: (i+1)*self.config.time_step]
                    for i in range(self.norm_data_test.shape[0]//self.config.time_step)]
            return np.array(test_x), label_data
        return np.array(test_x)

def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                  fmt='[ %(asctime)s ] %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save:
        file_handler = logging.FileHandler(config.log_save_path + "out.log")
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger

def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):

    label_data = origin_data.data_test[:, config.label_columns]
    #label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : ,
    #                                        config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_columns] + \
                   origin_data.mean[config.label_columns]   # 通过保存的均值和方差还原数据
    assert label_data.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_train_name[i] for i in config.label_columns]
    label_column_num = len(config.label_columns)

    # label 和 predict 是错开config.predict_length天的数据的
    # 下面是两种norm后的loss的计算方式，结果是一样的，可以简单手推一下
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_length:] - predict_norm_data[:-config.predict_length]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss = np.mean((label_data[config.predict_length:] - predict_data[:-config.predict_length] ) ** 2, axis=0)
    loss_norm = loss/(origin_data.std[config.label_columns] ** 2)
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    label_X = range(origin_data.data_test.shape[0] - origin_data.start_num_in_test)
    predict_X = [ x + config.predict_length for x in label_X]

    if not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
        for i in range(label_column_num):
            plt.figure(i+1)                     # 预测数据绘制
            plt.plot(label_X, label_data[:, i], label='label')
            plt.plot(predict_X, predict_data[:, i], label='predict')
            plt.title("Predict T-V {} ".format(label_name[i]))
            logger.info("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_length) +
                  str(np.squeeze(predict_data[-config.predict_length:, i])))
            if config.do_figure_save:
                plt.savefig(config.figure_save_path+"{}_predict_{}.png".format(config.continue_flag, label_name[i]))

        plt.show()

def save_prediction_data(config: Config, origin_data: Data, predict_norm_data: np.ndarray):

    # 通过保存的均值和方差还原数据
    predict_data = predict_norm_data * origin_data.std[config.label_columns] + \
                   origin_data.mean[config.label_columns]   
    
    predict_rows = predict_data.shape[0]
    
    init_data = pd.read_csv(config.test_data_path)
    test_data = init_data.values 
    test_row = test_data.shape[0]
   
    print("##INFO prediction shape: ", predict_data.shape, type(predict_data)) 
    print("##INFO test shape: ", test_data.shape, type(test_data)) 
    # TODO: prediction loss some data
    #result = np.concatenate((test_data[:(predict_rows-test_row)], predict_data), axis=1)
    test_data = test_data[:(predict_rows-test_row)]
    test_data[:, 2: 4] += predict_data.astype(int)
    
    
    if not os.path.exists(config.prediction_file_path):
        with open(config.prediction_file_path, 'w+') as f:
            pass

    df = pd.DataFrame(test_data[:, :4], columns=['line', 'trace', 'time', 'velocity'], dtype=int)
    df.to_csv(config.prediction_file_path, index=False, sep=',') 


def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data_gainer = Data(config)

        if config.do_train:
            train_X, train_Y = data_gainer.get_dataset('train')
            valid_X, valid_Y = data_gainer.get_dataset('val')
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict: 
            test_X, test_Y = data_gainer.get_test_data(return_label_data=True) 
            pred_result = predict(config, test_X)       # 这里输出的是未还原的归一化预测数据
            # TODO:save prediction result into csv file
            save_prediction_data(config, data_gainer, pred_result) 
            #draw(config, data_gainer, logger, pred_result)
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__=="__main__":
    import argparse
    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("-e", "--epoch", default=50, type=int, help="epochs num")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config

    main(con)
