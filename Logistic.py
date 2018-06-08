#!/usr/bin/python
# coding: utf8

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(file_name):
	dataMat=[]
	labelMat=[]
	fr=open(file_name)
	for line in fr.readlines():
		lineArr=line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat


def sigmoid(inX):
	return 2*1.0/(1+exp(-2*inX))-1

def gradAscent(dataMatIn,classLabels):
	dataMatrix=array(dataMatIn)
	num=classLabels.shape[0]
	labelMat=classLabels.reshape(num,1)
	m,n=shape(dataMatrix)
	alpha=0.001
	maxCycles=5000
	weights=ones((n,1))
	for k in range(maxCycles):
		h=sigmoid(dot(dataMatrix,weights))
		error=labelMat-h
		weights=weights+alpha*dot(dataMatrix.T,error)
	return weights

def stocGradAscent0(dataMatrix, classLabels):
    '''
    Desc:
        随机梯度下降，只使用一个样本点来更新回归系数
    Args:
        dataMatrix -- 输入数据的数据特征（除去最后一列）
        classLabels -- 输入数据的类别标签（最后一列数据）
    Returns:
        weights -- 得到的最佳回归系数
    '''
    m, n = shape(dataMatrix)
    alpha = 0.01
    # n*1的矩阵
    # 函数ones创建一个全1的数组
    weights = ones(n)  # 初始化长度为n的数组，元素全部为 1
    for i in range(m):
        # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(dataMatrix[i] * weights))
        # print 'dataMatrix[i]===', dataMatrix[i]
        # 计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
        error = classLabels[i] - h
        # 0.01*(1*1)*(1*n)
        # print weights, "*" * 10, dataMatrix[i], "*" * 10, error
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=200):
    '''
    Desc:
        改进版的随机梯度下降，使用随机的一个样本来更新回归系数
    Args:
        dataMatrix -- 输入数据的数据特征（除去最后一列数据）
        classLabels -- 输入数据的类别标签（最后一列数据）
        numIter=150 --  迭代次数
    Returns:
        weights -- 得到的最佳回归系数
    '''
    m, n = shape(dataMatrix)
    weights = ones(n)  # 创建与列数相同的矩阵的系数矩阵，所有的元素都是1
    # 随机梯度, 循环150,观察是否收敛
    for j in range(numIter):
        # [0, 1, 2 .. m-1]
        dataIndex = range(m)
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (
                1.0 + j + i
            ) + 0.0001  # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            # 随机产生一个 0～len()之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            randIndex = int(random.uniform(0, len(dataIndex)))
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            del (dataIndex[randIndex])
    return weights

def plotBestFit(dataArr,labelMat,weights):
	n=shape(dataArr)[0]
	xcord1=[]
	ycord1=[]
	xcord2=[]
	ycord2=[]
	for i in range(n):
		if labelMat[i]==1:
			xcord1.append(dataArr[i, 1])
			ycord1.append(dataArr[i, 2])
		else:
			xcord2.append(dataArr[i, 1])
			ycord2.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='>')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()


def simpleTest():
	dataMat,labelMat=loadDataSet("testSet.txt")
	dataArr=array(dataMat)
	labelArr=array(labelMat)
	weights=stocGradAscent1(dataArr,labelArr)
	plotBestFit(dataArr,labelMat,weights)

if __name__=="__main__":
	simpleTest()