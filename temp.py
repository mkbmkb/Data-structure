import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
'''
feap0 = np.load('/Users/makangbo/Desktop//feat_batch_0.npy')
feap1 = np.load('/Users/makangbo/Desktop/CBAM/features/feat_batch_1.npy')
feap2 = np.load('/Users/makangbo/Desktop/CBAM/features/feat_batch_2.npy')
feap3 = np.load('/Users/makangbo/Desktop/CBAM/features/feat_batch_3.npy')
feap4 = np.load('/Users/makangbo/Desktop/CBAM/features/feat_batch_4.npy')
'''

#y = np.load('/Users/makangbo/Desktop/n_feat_batch_2.npy')

x = np.load('/Users/makangbo/Desktop/fxxx.npy')

'''
n_feap1 = np.load('/Users/makangbo/Desktop/CBAM/features/n_feat_batch_1.npy')
n_feap2 = np.load('/Users/makangbo/Desktop/CBAM/features/n_feat_batch_2.npy')
n_feap3 = np.load('/Users/makangbo/Desktop/CBAM/features/n_feat_batch_3.npy')
n_feap4 = np.load('/Users/makangbo/Desktop/CBAM/features/n_feat_batch_4.npy')
'''

print(x.shape)
print('===========')

for i in range(13):
    x[0,0,:,:] = x[0,0,:,:] + x[0,i,:,:] 
    

#x = x.transpose(0,2,3,1)
plt.imshow(x[0,0,:,:]) 

    
#x = np.mean(feap0[0,:,:,:],axis = 0)
#print(x.shape)


#fig, ax = plt.subplots(16, 16, figsize=(32,32))

'''
num = 0
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        ax[i,j].imshow(n_feap0[0,i*16+j,:,:].clip(0, 255))
        num += 1
plt.show()
'''




'''
plt.figure(figsize=(32, 32))
for i in range(4):
    x = np.mean(feap0[i,:,:,:],axis = 0)
    plt.subplot(10,4,i+1)
    plt.imshow(x)
    
for i in range(4):
    x = np.mean(feap1[i,:,:,:],axis = 0)
    plt.subplot(10,4,i+5)
    plt.imshow(x)    

for i in range(4):
    x = np.mean(feap2[i,:,:,:],axis = 0)
    plt.subplot(10,4,i+9)
    plt.imshow(x)

for i in range(4):
    x = np.mean(feap3[i,:,:,:],axis = 0)
    plt.subplot(10,4,i+13)
    plt.imshow(x)
    
for i in range(4):
    x = np.mean(feap4[i,:,:,:],axis = 0)
    plt.subplot(10,4,i+17)
    plt.imshow(x)


for i in range(4):
    x = np.mean(n_feap0[i,:,:,:],axis = 0)
    plt.subplot(10,4,i+21)
    plt.imshow(x)

for i in range(4):
    x = np.mean(n_feap1[i,:,:,:],axis = 0)
    plt.subplot(10,4,i+25)
    plt.imshow(x)    

for i in range(4):
    x = np.mean(n_feap2[i,:,:,:],axis = 0)
    plt.subplot(10,4,i+29)
    plt.imshow(x)

for i in range(4):
    x = np.mean(n_feap3[i,:,:,:],axis = 0)
    plt.subplot(10,4,i+33)
    plt.imshow(x)
    
for i in range(4):
    x = np.mean(n_feap4[i,:,:,:],axis = 0)
    plt.subplot(10,4,i+37)
    plt.imshow(x)
'''