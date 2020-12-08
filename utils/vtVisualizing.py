import cv2
import os
from readFromDat import getSpecCMPData
from tqdm import tqdm





'''
函数的作用是将速度和时间的Ground Truth 转换为图片上的坐标值。
minW:速度标尺的最小值。
maxW:速度标尺的最大值
minH:时间标尺的最小值
maxH:时间标尺的最大值
img_h:图片的高
img_w:图片的宽
gt_v:ground truth的速度值
gt_h:ground truth的时间
'''

def VT2point(minW,minH,maxW,maxH,img_w,img_h,gt_v,gt_t):
    point_x=((gt_v-minW)/(maxW-minW))*img_w
    point_y=((gt_t-minH)/(maxH-minH))*img_h
    return point_x ,point_y

def VTlist2Pointlist(data,img_path):
    img=cv2.imread(img_path)
    img_h,img_w,_=img.shape
    length=len(data)
    Pointlist=[]
    for i in range(0,length):
        list = []
        str = data[i].split(',')
        #print(str)
        gt_v = int(str[3])
        gt_t = int(str[2])
        point_x, point_y = VT2point(min_v, min_t, max_v, max_t, img_w, img_h, gt_v, gt_t)
        list.append(point_x)
        list.append(point_y)
        Pointlist.append(list)
    return Pointlist


'''
  函数的作用是将【谱线、速度时间对】可视化到速度谱上。
  pointlist：速度谱上要画出点的列表。
             ex: { {x_1,y_1},{x_1,y_2} ,...  }
  img：opencv读取的图片矩阵。
  desPath:可视化图片的保存地址。
  ps: opencv不懂的部分请查看opencv文档：
      https://www.kancloud.cn/aollo/aolloopencv/260982
'''
def drawLineOnImage(pointlist, img, desPath, clr_config):
    
    # load the line and point color configuration
    line_color = clr_config['line_color']  # point color 
    line_thickness = clr_config['line_thick']  # line thickness
    line_type = clr_config['line_type']   
   
    pt_color = clr_config['pt_color']
    pt_thickness = clr_config['pt_thick']
    
    length=len(pointlist)
    for i in range(1,length):
        ptStart = (int(pointlist[i-1][0]), int(pointlist[i-1][1]))
        ptEnd = (int(pointlist[i][0]), int(pointlist[i][1]))
        #print(ptStart,ptEnd)
        img=cv2.line(img, ptStart, ptEnd, line_color, line_thickness, line_type)
        img = cv2.circle(img, ptStart, 4, color=pt_color, thickness=pt_thickness)
    ptEnd = (int(pointlist[length-1][0]), int(pointlist[length-1][1]))
    img = cv2.circle(img, ptEnd, 4, color=pt_color, thickness=pt_thickness)
    cv2.imwrite(desPath, img)



'''
可视化速度谱的标签。
data：速度谱标签的文本文件。
path: 读取速度谱图片的地址
despath:保存可视化结果的地址。
'''
def vis_dir(label_file ,img_path, desPath, clr_config):
    img_list=os.listdir(img_path)

    if not os.path.exists(desPath):
        os.mkdir(desPath)

    for file in tqdm(img_list):
        #print(file)
        # the line and cmp format can be different with different dataset, it should be modified respectively.
        if file.endswith('.png'):
            str = file.split('_')
            #linenum = int(str[5]) # for dq8 name format is: v33_velocity_AI_dq8_energy_250_2050.png
            #cmpnum = int(str[6][:-4]) # for dq8

            linenum = int(str[4]) # for hade, which name format is :v33_velocity_hade_energy_2280_2080.png
            cmpnum = int(str[5][:-4])
            data = getSpecCMPData(linenum, cmpnum, label_file)
            img_file = os.path.join(img_path, file)
            if len(data) == 0:
                continue
            Pointlist = VTlist2Pointlist(data, img_file) 
            img = cv2.imread(img_file)
            drawLineOnImage(Pointlist, img, os.path.join(desPath, file), clr_config)



if __name__=="__main__":
    # feat: may hade image and vt shape
    min_v=1300
    min_t=0
    max_v=5500
    max_t=7000
    
    # dq8 data rage
    #min_v=1200
    #min_t=0
    #max_v=7000
    #max_t=8000
    
    # line and color configuration, BGR version for opencv

    # GT config ,color: dark brown
    color_config_gt = {'line_color':(79, 79, 79), 'line_thick':2, 'line_type':4 , 'pt_color':(79, 79, 79), 'pt_thick':3}
    # fcos prediction config : bright green  
    color_config_fcos = {'line_color':(0, 255, 127), 'line_thick':2, 'line_type':4 , 'pt_color':(0, 255, 127), 'pt_thick':3}
    # lstm prediction config, color Turquoise  
    #color_config_lstm = {'line_color':(238, 229, 142), 'line_thick':2, 'line_type':4 , 'pt_color':(238, 229, 142), 'pt_thick':3}
    # orange
    color_config_lstm = {'line_color':(0, 165, 255), 'line_thick':2, 'line_type':4 , 'pt_color':(0, 165, 255), 'pt_thick':3}
    
    # path where these results visualized saved
    save_path = './dataTest/pred_visual'
    #img_path = './dataTest/test-complex'
    img_path = './dataTest/test-real' # image path for hade
    #save_path_fcos = './dataTest/fcos_lstm_dq8_visual'
    save_path_fcos = './dataTest/fcos_hade_visual'

    # v-t data file path 
    # label file for dq8AI
    #label_file_gt = './dataTest/complex_test_label2.csv'
    #label_file_fcos = './dataTest/complex_test_withFCOS.csv' 
    #label_file_lstm = '../data/complex_test_prediction.csv' 
    #label_file_lstm = './dataTest/complex_test_predicted_data.csv'
    #label files for hade
    label_file_gt = './dataTest/data_test_hade_label.csv'
    label_file_fcos = './dataTest/data_test_hade_FCOS.csv' 
    #label_file_lstm = '../data/complex_test_prediction.csv' 
    
    # 
    vis_dir(label_file_gt, img_path, save_path_fcos, color_config_gt)
    vis_dir(label_file_fcos, save_path_fcos, save_path_fcos, color_config_fcos)
    #vis_dir(label_file_lstm, save_path_fcos, save_path_fcos, color_config_lstm)
    


