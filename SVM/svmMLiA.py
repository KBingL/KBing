
import random
import  numpy  as np
import matplotlib.pyplot as plt

 
def loadDataSet(fileName):
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat,labelMat

def selectJrand(i,m):		#从0~m个alpha中选择除i外的另一个alpha
	j = i
	while j == i:
		j = int(random.uniform(0,m))
	return j
 
def clipAlpha(aj,H,L):		#用于调整大于H或小于L的值
	if aj>H:
		aj = H
	if L>aj:
		aj = L
	return aj

def order1():
    dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')
    b,alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    
"""
创建一个alpha向量并将其初始化为O向量 当迭代次数小于最大迭代次数时（外循环）
 对数据集中的每个数据向量（内循环） ：

如果该数据向量可以被优化： 随机选择另外一个数据向量 
同时优化这两个向量 如果两个向量都不能被优化，退出内循环 
如果所有向量都没被优化，增加迭代数目，继续下一次循环

该函数有5个输人参数，分别 是 ：数据集、类别标签、常数0 、容错率和取消前最大的循环次数

"""
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn); 
    labelMat = np.mat(classLabels).transpose()
    b = 0; 
    m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter_num = 0
     # alphaPairsChanged 用来更新的次数
    # 当遍历  连续无更新 maxIter 轮，则认为收敛，迭代结束
    while (iter_num < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            #注意一
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: 
                    print("L==H"); 
                    continue
                #注意二
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: 
                    print("eta>=0"); 
                    continue
                
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
              
                alphas[j] = clipAlpha(alphas[j],H,L)
               
                if (abs(alphas[j] - alphaJold) < 0.00001): print("alpha_j变化小，不需要更新"); continue
                #注意三
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
 
                #注意四
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
             
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
               
                alphaPairsChanged += 1
              
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num,i,alphaPairsChanged))
       
        if (alphaPairsChanged == 0): 
            iter_num += 1
        else: iter_num = 0
        print("迭代次数: %d" % iter_num)
    #注意五
    return b,alphas

def calcWs(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()
 
def showClassifer(dataMat,labelMat,alphas, w, b):
    data_plus = []                                  
    data_minus = []
    #注意六
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              
    data_minus_np = np.array(data_minus)            
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) 
    #注意七
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #注意八
    for i, alpha in enumerate(alphas):
        if 0.6>abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
        if 0.6==abs(alpha) :
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='yellow')
    plt.show()

"""
解释器提示如：SyntaxError: invalid character in identifier，　
但又一直找不到问题点的话，请确保代码行内没有夹杂中文的空格，tab等，非文字字符．
"""

def order2():
    #shape(alphas[alphas>0])
    for i in range(100):
        if alphas[i]>0.0:
            print(dataArr[i],labelArr[i])
    
   



