from numpy import *
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


 
def selectJrand(i,m): 
    j=i
    while (j==i):
        j=int(random.uniform(0,m))
    return j
 
def clipAlpha(aj,H,L):  
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj
 
def kernelTrans(X, A, kTup): 
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': 
        K = X * A.T
    elif kTup[0]=='rbf': 
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) 
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K
 
 
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  
        self.X = dataMatIn  
        self.labelMat = classLabels 
        self.C = C 
        self.tol = toler 
        self.m = shape(dataMatIn)[0] 
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0 
        self.eCache = mat(zeros((self.m,2))) 
        self.K = mat(zeros((self.m,self.m))) 
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
 
 
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
 
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]  
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE): 
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej
 
 
def updateEk(oS, k): 
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
 
 
def innerL(i, oS):
    Ei = calcEk(oS, i) 
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)): 
        j,Ej = selectJ(i, oS, Ei) 
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] 
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta 
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L) 
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < oS.tol): 
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i) 
       
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]<oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0
 
 
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('rbf', 1.3)):    
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False 
        elif (alphaPairsChanged == 0): entireSet = True  
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas

def calculateW(alphas,dataArr,labelArr):
    alphas, dataMat, labelMat = array(alphas), array(dataArr), array(labelArr)
    sum = 0
    for i in range(shape(dataMat)[0]):
        sum += alphas[i]*labelMat[i]*dataMat[i].T
    print(sum)
    return sum
"""
通过上一个函数返回的alpha，b，通过拉格朗日法推出的w公式，计算w，参考
"""
"""
    Function：   利用核函数进行分类的径向基测试函数

    Input：      k1：径向基函数的速度参数

    Output： 输出打印信息
""" 

def testRbf(k1=1.3):
    
	dataArr, labelArr = loadDataSet('testSetRBF.txt')
	b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf',k1))
	datMat = mat(dataArr)
    #初始化数据矩阵和标签向量
	labelMat = mat(labelArr).transpose()
    #记录支持向量序号
	svInd = nonzero(alphas.A>0)[0]		#值大于0的参数alpha的数量
    #读取支持向量
	sVs = datMat[svInd]		#构建支持向量矩阵
    #读取支持向量对应标签
	labelSV = labelMat[svInd]	#支持向量对应的分类
	print ("there are %d Support Vectors" % shape(sVs)[0])
    #获取数据集行列值
	m,n = shape(datMat)
    #初始化误差计数
	errorCount = 0
    #遍历每一行，利用核函数对训练集进行分类
	for i in range(m):
        #利用核函数转换数据
		kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',k1))		#计算数据集中第i个样本与样本集合的kernel结果
        #仅用支持向量预测分类
		predict = kernelEval.T * multiply(labelSV,alphas[svInd])+b 		#计算第i个样本的预测结果
        #预测分类结果与标签不符则错误计数加一
		if sign(predict)!=sign(labelArr[i]):
			errorCount = errorCount+1
	print ("the training error rate is: %f" %(float(errorCount)/m)	)#先计算训练误差
	dataArr,labelArr = loadDataSet('testSetRBF2.txt')
	errorCount = 0
	datMat = mat(dataArr)
	labelMat = mat(labelArr).transpose()
	m,n = shape(datMat)
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',k1))
		predict = kernelEval.T * multiply(labelSV,alphas[svInd])+b
		if sign(predict)!=sign(labelArr[i]):
			errorCount = errorCount+1
	print ("the test error rate is: %f" %(float(errorCount)/m))	#再计算测试误差


def showClassifer(dataMat,labelMat,alphas):
    data_plus = []                                  
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = array(data_plus)              
    data_minus_np = array(data_minus)            
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30, alpha=0.7)   
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1], s=30, alpha=0.7) 
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()
    
"""
手写识别
"""  
def img2vector(filename):			#从图像转换为向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
 
def loadImages(dirName):		#载入图片，返回图片的数据矩阵和分类
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)		#如果是9就是负类，另外的数字是正类，因为svm是二分类算法
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    
 
def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 				#组成支持向量的矩阵
    labelSV = labelMat[svInd]		#支持向量的分类
    print ("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)		#计算支持向量与第i个样本之间的核函数结果
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b 		#计算第i个样本的预测结果，只需要算与支持向量之间的结果，因为其他样本的alpha都为0
        if sign(predict)!=sign(labelArr[i]): 
        	errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print ("the test error rate is: %f" % (float(errorCount)/m))
    
def predict():
    
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf',k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd])+b
    
    


