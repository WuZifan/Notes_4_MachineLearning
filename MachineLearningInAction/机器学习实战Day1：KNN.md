# 机器学习实战Day1：KNN

标签（空格分隔）： MachineLearning

---
####**0. 机器学习术语普及**
* 1）在监督学习中，用来判断分类的属性称之为**特征**，分类的结果则称之为**目标变量**。
![1_特征和目标变量.jpg-42.7kB][1]

* 2）**数值型变量**指取值连续的变量，比如100,101.1等等；**标称型标量**指取值有一定范围的枚举型变量。

* 3）**监督学习和非监督学习的适用范围**：![2_监督和无监督.jpg-50.4kB][2]

* 4）开发的基本步骤：收集数据->准备输入数据->分析输入数据(optional)->训练算法->测试算法->使用算法


####**1. 什么是KNN**
KNN：
    
    又叫K-邻近算法，是一种监督学习算法。
    优点：精度高，对异常值不敏感，无数据输入假定
    缺点：计算复杂度高，空间复杂度高
    适用数据：数值型，标称型

算法自然语言描述：

    1.存在一个样本集，也叫训练集，通常用矩阵表示；矩阵中每一行表示一个实例，每个实例都有标签。
    2.对于新输入没有标签的数据，计算它与训练集中每一个实例的距离，并选出距离最近的k个实例。
    3.对于这k个实例，出现次数最多标签，就是这个新数据的预测值。

一般流程描述：
![3_k邻近一般流程.jpg-26.3kB][3]

####**2. ToyExample**
这里是简单演示一下KNN，训练集表示如下：
![4_knnToy.jpg-8.3kB][4]
一共4个样本，2种特征。样本被分为A,B两类。

#####**2.1 数据准备**
在这个例子中，数据由自己创建，为了方便计算，数据格式我们选用numpy提供的**‘array’**格式来存储。该格式可以方便的进行矩阵相关的计算。
``` PY
# -*- coding=utf-8 -*-
from numpy import *
import operator # 排序时候用的
from os import listdir

# 注意类型，一个是array类型的，一个就只是单纯的list类型
def createDataSet():
    group = array([[1.0,1.1],
                  [1.0,1.0],
                  [0,0],
                  [0,0.1]])

    labels=['A','A','B','B']
    return group,labels
```
#####**2.2 分类器**
这里的分类器干了下面这些事情：

    1.计算新输入数据和训练集中每一个数据之间的距离，并保存。
    2.按距离从小到大升序排序。
    3.取出前k个数据。
    4.统计前k个数据中，每一个label出现的次数，按照降序排序。
    5.输出出现次数最高的label。
    
``` PY
# 简单的分类器
def classify0(inX,dataSet,labels,k):
    # 拿到输入的训练array的长度
    dataSetSize=dataSet.shape[0]
    
    # inX表示输入，[0,0]
    # tile是将inX重复dataSetSie遍，并且形成（dataSetSize，1）这样大小的一个矩阵
    # 注意技巧，这里是把输入数据复制成一个和训练集一样大小的矩阵
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    
    # 计算每一项差的平方
    sqDiffMat=diffMat**2
    # print(sqDiffMat)
    
    # 对所有项的差平方求和，得到距离的平方
    sqDistances=sqDiffMat.sum(axis=1)
    # print(sqDistances)
    
    # 对距离的平方开根号，得到距离
    distances=sqDistances**0.5
    # print(distances)
    
    # 对距离排序,输出结果以下标作为结果
    sortedDistIndicies=distances.argsort()
    # print(sortedDistIndicies)
    
    classCount={}
    
    # print('---------------')
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        # classCount是一个字典，后面的get表示拿出名为voteIlabel对应的内容，没有的话就写0
        # 之后把该内容加1，并且以voteIlabel为key，存储到字典中
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
        # print(classCount)
    
    # 最后按照字典中的value项进行递增排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    
    # 返回k个结果中出现次数最多的那一类型
    return sortedClassCount[0][0]
```

####**3. 约会网站**
目标：

    每个人有三种不同的属性，对于Helen，她希望根据这三种属性判断下个男性是否对她有吸引力。
    她之前已经有标签好的数据。

#####**3.0 基本流程**
![5_约会网站.jpg-32.6kB][5]
#####**3.1 从文件中加载数据**
``` PY
# 加载文件中的数据，并把它分为特征矩阵和labels
def file2matrix(filename):
    # 打开文件:
    fr=open(filename)
    # 将文件变成一个list，每一行为一项
    arrayOLines=fr.readlines()
    # print(arrayOLines)
    # 计算有多少行
    numberOfLines=len(arrayOLines)
    # 构建一个大小为行数*3的矩阵--我觉得这里3写死不太好
    returnMat=zeros((numberOfLines,3))
    # 构建label的容器list，方便存储
    classLabelVector=[]
    myLabel={}
    labelIndex=0;
    # 下标
    index=0
    # 对每一行数据
    for line in arrayOLines:
        # 去除头尾空格
        line=line.strip()
        # 以'\t'分割字符串，返回一个list
        listFromLine=line.split('\t')
        # 矩阵中第index行中的每一列（4列），被listFromLine的0到3（4列）赋值
        returnMat[index,:]=listFromLine[0:3]
        # label容器的值等于list的最后一项
        classLabelVector.append(listFromLine[-1])
        if myLabel.get(listFromLine[-1],0)==0:
            myLabel[listFromLine[-1]]=labelIndex;
            labelIndex+=1
        # index加一
        index+=1
    # print(myLabel.keys())
    return returnMat,classLabelVector
```
#####**3.2 归一化特征**
为什么要归一化/正则化：
![6_正则化.jpg-38.9kB][6]

``` PY
# 归一化
def autoNorm(dataSet):
    # 得到每一列的最大最小值
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    # 计算每一列的取值范围
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    # 得到有多少行
    m=dataSet.shape[0]
    # 把原来矩阵中每一个数据都减去当前列的最小值
    normDataSet=dataSet-tile(minVals,(m,1))
    # 把上面计算得到的结果矩阵除以范围
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
```

#####**3.3 分类器**
用的是和2.2中同一个分类器

#####**3.4 测试**
测试选用训练集中10%的数据作为测试集，但是这样也会有误判的情况，我理解的原因如下：

    # 这里说一下为什么会有错误值，哪怕这10%的数据也被用作判断
    # 是因为，我们取k=3，对数据A，对应标签为A'，它在整个数据集中，那么显然会有一行，
    # 令其与A的距离为0，令这个数据被判定为A’
    # 但是，如果有两个其他也很近，但是标签为B'
    # 而这样，这个数据就被误判了。
    
``` PY
# 分类器测试
def datingClassTest():
    # 取10%的作为测试数据
    hoRatio=0.1
    # 加载文件，格式化数据
    datingDataMat,datingLables=file2matrix('datingTestSet.txt')
    # 归一化数据
    normMat,ranges,minVals=autoNorm(datingDataMat)
    # 得到有多少行
    m=normMat.shape[0]
    # 取出测试用例
    numTestVecs=int(m*hoRatio)
    # 统计有多少个不靠谱
    errorCount=0.0
    # 这里说一下为什么会有错误值，哪怕这10%的数据也被用作判断
    # 是因为，我们取k=3，对数据A，对应标签为A'，它在整个数据集中，那么显然会有一行，
    # 令其与A的距离为0，令这个数据被判定为A’
    # 但是，如果有两个其他也很近，但是标签为B'
    # 而这样，这个数据就被误判了。
    for i in range(numTestVecs):
        # 用分类器计算结果
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLables[numTestVecs:m],3)
        print('predict: '+ classifierResult+' realResult: '+ datingLables[i])
        # 累加错误值
        if (classifierResult!=datingLables[i]):
            errorCount+=1.0
    print('error rate'+str((errorCount/float(numTestVecs))))
```

#####**3.5 实际使用**
这里别忘了对已有数据进行正则化。

另外，我觉得这里将新数据直接减去训练集中的最小值不是很妥当，万一新数据会带来新的最小值呢？？不过这一误差会带来怎样的影响，还是不能确定。
``` PY
# 加入命令行输入
def classifyPerson():
    # python3的 输入变成了input()
    percentTats=float(input('Video games: '))
    ffMiles=float(input('filer miles: '))
    iceCream=float(input('icecream '))
    # 将已有数据提炼为可以使用的格式
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    # 将数据正则化
    norMat,ranges,minVals=autoNorm(datingDataMat)
    # 将输入数据转化为numpy的array格式
    inArr=array([ffMiles,percentTats,iceCream])
    # 将输入数据（同样要正则化），格式化&正则化后的已有数据，标签以及k值输入分类器
    classifierResult=classify0((inArr-minVals)/ranges,norMat,datingLabels,3)
    # 打印返回结果
    print(classifierResult)
```

####**4.手写字识别**
![7_手写字.jpg-35kB][7]
手写字以这样的文本进行存储，每个文本表示一个手写字，训练集由2000个文本，每个数字大约200个文本。
每个文本是32*32大小的。

#####**4.1 准备数据**
这里主要涉及到批量获取文件名，以及循环加载文件中的内容，并把文件中32*32的数据转化为1*1024的数据
``` PY
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
```

#####**4.2 识别**
用的还是2.2中的分类器。可见只要数据格式合适，分类器基本通用。
``` PY
# 识别手写字
def handwritingClassTest():
    hwLabels=[]
    # 获取文件夹下所有的文件名
    trainingFileList=listdir('trainingDigits')
    # 获取有多少个训练文件
    m=len(trainingFileList)
    # 构造一个训练矩阵，里面每一行都是一个实例
    trainingMat=zeros((m,1024))
    # 对于每一个实例
    for i in range(m):
        # 提取文件名（带扩展名）
        fileNameStr=trainingFileList[i]
        # 提取文件名（删除了扩展名）
        fileStr=fileNameStr.split('.')[0]
        # 提取文件名中‘_’前面的信息
        classNumStr=int(fileStr.split('_')[0])
        # 将上一步的信息作为目标变量存储
        hwLabels.append(classNumStr)
        # 将格式化后的数据存储到训练矩阵中
        trainingMat[i,:]=img2vector('trainingDigits/'+fileNameStr)
    # 对测试数据集做差不多同样的事
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        # 得到一个测试数据的格式化后的数据
        vectorUnderTest=img2vector('testDigits/'+fileNameStr)
        # 对其进行分类
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        # 判断正确与否。
        print('class Result: '+str(classifierResult)+' the real: '+ str(classNumStr))
        if classifierResult!=classNumStr:errorCount+=1
    print('error: '+str(errorCount)+' errorRate: '+ str(float(errorCount/mTest)))
```

####**5.总结**
步骤：

    1.准备数据，数据格式化。
    2.数据归一化/正则化
    3.得到新输入数据，新输入数据正则化。
    4.计算距离，排序。
    5.取前k个值，计算label出现次数。
    6.取出现次数最多的label作为输出。


  [1]: http://static.zybuluo.com/w460461339/1w0wofape7wppncn7bb5mjo2/1_%E7%89%B9%E5%BE%81%E5%92%8C%E7%9B%AE%E6%A0%87%E5%8F%98%E9%87%8F.jpg
  [2]: http://static.zybuluo.com/w460461339/uclm9pf0ah8rlattmnlvotnp/2_%E7%9B%91%E7%9D%A3%E5%92%8C%E6%97%A0%E7%9B%91%E7%9D%A3.jpg
  [3]: http://static.zybuluo.com/w460461339/e1t4kn6rwmkujpiuvjzx3n8n/3_k%E9%82%BB%E8%BF%91%E4%B8%80%E8%88%AC%E6%B5%81%E7%A8%8B.jpg
  [4]: http://static.zybuluo.com/w460461339/xohs8kkall1p3prn1x82m8jk/4_knnToy.jpg
  [5]: http://static.zybuluo.com/w460461339/l47hu6av9paregli1yx9j3bi/5_%E7%BA%A6%E4%BC%9A%E7%BD%91%E7%AB%99.jpg
  [6]: http://static.zybuluo.com/w460461339/zkopez12emr3feec2lqxsi78/6_%E6%AD%A3%E5%88%99%E5%8C%96.jpg
  [7]: http://static.zybuluo.com/w460461339/bdlnntzgaadv5z5jo2ubha92/7_%E6%89%8B%E5%86%99%E5%AD%97.jpg