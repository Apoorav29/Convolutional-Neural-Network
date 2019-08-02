import imageio
import numpy as np


im = imageio.imread('img1.png')
im1 = im[:,:,0]

def max_func(a, b, c, d):
	lst = [a,b,c,d]
	lst = sorted(lst, reverse=True)
	return lst[0]

def sigmoid_func(mat):
	return 1/(1+np.exp(-mat))

def softmax(mat):
	return np.exp(mat)/np.sum(np.exp(mat))

#random filters
filt_arr = []
conv1_arr=[]
sub1_arr = []
for i in range(6):
	filt_arr.append(np.random.rand(5,5))
	conv1_arr.append(np.zeros((28,28)))
	sub1_arr.append(np.zeros((14,14)))

for i in range(6):
	for j in range(28):
		for k in range(28):
			sum_val=0.0
			for l in range(5):
				for m in range(5):
					sum_val+=im1[j+l][k+m]*filt_arr[i][l][m]
			sum_val/=filt_arr[i].sum()
			conv1_arr[i][j][k]=sum_val

for i in range(6):
	for j in range(14):
		for k in range(14):
			temp_val = max_func(conv1_arr[i][2*j][2*k], conv1_arr[i][2*j+1][2*k], conv1_arr[i][2*j][2*k+1], conv1_arr[i][2*j+1][2*k+1])
			sub1_arr[i][j][k]=temp_val

filt2_arr = []
conv2_arr = []
sub2_arr = []

for i in range(16):
	filt2_arr.append(np.random.rand(6,5,5))
	conv2_arr.append(np.zeros((10,10)))
	sub2_arr.append(np.zeros((5,5)))

for i in range(16):
	for j in range(10):
		for k in range(10):
			sum_val=0.0
			for l in range(5):
				for m in range(5):
					for z in range(6):
						sum_val+=sub1_arr[z][j+l][k+m]*filt2_arr[i][z][l][m]
			sum_val/=filt2_arr[i].sum()
			conv2_arr[i][j][k]=sum_val

# print(conv2_arr[0].shape)
for i in range(16):
	for j in range(5):
		for k in range(5):
			temp_val = max_func(conv2_arr[i][2*j][2*k], conv2_arr[i][2*j+1][2*k], conv2_arr[i][2*j][2*k+1], conv2_arr[i][2*j+1][2*k+1])
			sub2_arr[i][j][k]=temp_val

sub2_arr=np.array(sub2_arr)
inp = np.zeros(400)
inp = sub2_arr.flatten(order='C')
# print(inp)
theta_arr = []
out_arr = []

nodes = [400, 120, 84, 10]

for i in range(4):
	out_arr.append(np.zeros(nodes[i]))
for i in range(len(nodes)-2):
	theta_arr.append(np.random.rand(nodes[i], nodes[i+1]) * 0.01)
theta_arr.append(np.random.rand(84,10)-0.5)
out_arr[0]=inp
# print(inp)
for i in range(len(nodes)-2):
	out_arr[i+1]=sigmoid_func(np.dot(np.transpose(theta_arr[i]), out_arr[i]))
out_arr[3]=softmax(np.dot(np.transpose(theta_arr[2]), out_arr[2]))
# print(out_arr[3].shape)
print(out_arr[3])