# -*- coding:utf-8 -*-

import numpy as np

def load(filename,length,start,amount):
	features = [[0 for i in range(length)] for j in range(0,amount)]
	result = [[0] for j in range(amount)]
	file = open(filename)
	end = start + amount
	for i in range(start, end):
		line = file.readline()
		value = line.split()
		for j in range(length):
			features[i-start][j] = float(value[j])
		result[i-start][0] = float(value[-1])
		
	output1 = np.array(features)
	output2 = np.array(result)
	return output1,output2,end


#non linear function
def sigmoid(X,derive = False):
	if not derive:
		return 1/(1 + np.exp(-X))
	else:
		return X * (1 - X)

def relu(X,derive = False):
	if not derive:
		return np.maximum(0,X)
	else:
		return (X>0).astype(float)
	
nonline = relu
#nonline = sigmoid
		
def predict(features,result,W1,W2,b1):
	#layer 1
	A1 = np.dot(features, W1) + b1
	Z1 = nonline(A1)
	
	#layer 2
	A2 = np.dot(Z1,W2)
	_y2 = Z2 = nonline(A2)
	
	print "-------------"
	print _y2
	
	for i in range(np.shape(_y2)[0]):
		if _y2[i][0] < 0.5:
			temp = 0
		else:
			temp = 1
		
		if (temp == result[i][0]):
			print True
		else:
			print False
			
def test():
	print test
	
def main():	
	#input
	X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
	#output
	y = np.array([[0],[1],[1],[0]])
	
	queue_length = 3
	train_data_amount = 4
	
	'''
	train_data = "test_data.txt"
	queue_length = 3
	train_data_amount = 1#10
	X,Y,temp = load(train_data,queue_length,0,train_data_amount)
#	print X
	'''
	#weights,bias
	np.random.seed(1)
	#随机生成初始权重
	W1 = 2 * np.random.random((queue_length,train_data_amount)) - 1
	W2 = 2 * np.random.random((train_data_amount,1)) - 1
	b1 = 0.1 * np.ones((train_data_amount,))
	b2 = 0.1 * np.ones((1,))

	train_times = 600
	
	for i in range(train_times):
		#layer 1
		A1 = np.dot(X, W1) + b1
		Z1 = nonline(A1)

		#layer 2
		A2 = np.dot(Z1,W2)
		_y = Z2 = nonline(A2)
		
		cost = _y - y
		#cost = (y - _y)^2 /2
		print "Error:\t",
		print format(np.mean(np.abs(cost)))
		
		#calculte delta
		delta_A2 = cost * nonline(Z2,derive = True)
		delta_b2 = delta_A2.sum(axis = 0)
		delta_W2 = np.dot(Z1.T,delta_A2)
		
		delta_A1 = np.dot(delta_A2,W2.T) * nonline(Z1,derive=True)
		delta_b1 = delta_A1.sum(axis = 0)
		delta_W1 = np.dot(X.T,delta_A1)
		
		#Apply deltas
		rate = 0.1
		W1 -= rate * delta_W1
		b1 -= rate * delta_b1
		W2 -= rate * delta_W2		
		b2 -= rate * delta_b2
		
	else:
		print "predict:\t"
		print _y
	
	end = 0
	test_data = "test_data.txt"
	while(end < 4):
		test_data_features,test_data_results,end  = load(test_data,queue_length,end,train_data_amount)
		print "end:\t%d" % end 
		print test_data_features
		predict(test_data_features,test_data_results,W1,W2,b1)

	X2 = np.array([ [1,1,1],[0,0,1],[0,1,1],[1,0,1] ])
	Y2 = np.array([[0],[0],[1],[1]])

main()