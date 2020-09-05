'''
function: read tvs with respect to line-trace and format
'''
def list2str(list):
    length=len(list)
    str=''
    for i in range(length):
        str+=list[i]+','
    str = str[:-1]
    return str

def parseStr(str):
    str=str.split(' ')
    list1 = [x for x in str if x.strip()]
    list2=list2str(list1)
    return list2

# Read txt file and converte into string list which one string per line and space removed
def getdata2list(datapath):
    data = []
    # data = '/Users/hongqiangwang/Desktop/data.txt'
    f = open(datapath, "r")  # 设置文件对象
    line = f.readline()
    line = line[:-1]
    if line != '':
        data.append(parseStr(line))
    while line:  # 直到读取完文件
        line = f.readline()  # 读取一行文件，包括换行符
        line = line[:-1]  # 去掉换行符，也可以不去
        # print(line)
        if line != '':
            data.append(parseStr(line))
    f.close()  # 关闭文件
    return data

# 获取特定的线号和CMP道集的数据
def getSpecCMPData(linenum,cmp_num,datapath):
    data = []
    f = open(datapath, "r")  # 设置文件对象
    line = f.readline()
    if str(cmp_num) in line :
        line = line[:-1]
        if line != ''and str(linenum)in line:
            templine=parseStr(line)
            str2=templine.split(',')
            if str(cmp_num) == str2[1] and str(linenum)==str2[0]:
                data.append(parseStr(templine))
    while line:  # 直到读取完文件
        line = f.readline()  # 读取一行文件，包括换行符
        line = line[:-1]  # 去掉换行符，也可以不去
        if line != '' and str(cmp_num)  in line and str(linenum)in line:
            templine=parseStr(line)
            str2=templine.split(',')
            if str(cmp_num) == str2[1] and str(linenum)==str2[0]:
                data.append(parseStr(templine))
    f.close()  # 关闭文件
    return data

data=getSpecCMPData(970, 1650,'./dataTest/label.txt')

print(len(data) == 0)
#tmp = data[0]
