import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from h5py import File
import os
import scipy.io
import statistics
import random
from six.moves import xrange
import fnmatch
import cv2
import Mytransforms_JHMDB as Mytransforms_JHMDB

# borrowed from: https://github.com/lawy623/LSTM_Pose_Machines/blob/master/dataset/JHMDB/JHMDB_PreData.m
# order
# 0: neck    1:belly   2: face
# 3: right shoulder  4: left shoulder
# 5: right hip       6: left hip
# 7: right elbow     8: left elbow
# 9: right knee      10: left knee
# 11: right wrist    12: left wrist
# 13: right ankle    14: left ankle

def get_train_test_annotation(dataRoot):
    
    #need
    subFolder = os.path.join(dataRoot, 'sub_splits')
    imageFolder = os.path.join(dataRoot, 'Rename_Images')
    poseFolder = os.path.join(dataRoot, 'joint_positions')
    
    #/Users/makangbo/Desktop/Sub-JHMDB/Rename_Images

    #/Users/makangbo/Desktop/Sub-JHMDB/joint_positions

    #/Users/makangbo/Desktop/Sub-JHMDB/puppet_mask
    
    maskFolder = os.path.join(dataRoot, 'puppet_mask')
    
    # baselineFolder = os.path.join(dataRoot, 'baseline_resnet101')
    totTXTlist = os.listdir(subFolder)

    trainAnnot = []
    testAnnot = []
    for i in range(0, len(totTXTlist)):
        filename = os.path.join(subFolder, totTXTlist[i])
        action = totTXTlist[i].split('_test_')[0]
        
        
        with open(filename) as f:
            content = f.readlines()

        for t in range(0, len(content)):
            
            folder_to_use = content[t].split('\n')[0].split('.avi')[0]
            traintest = content[t].split('\n')[0].split('.avi')[1]   # 1: train; 2: test
            
            #HP_PRISONER_OF_AZKABAN_walk_f_cm_np1_ri_med_2.avi 2
            #=====
            #HP_PRISONER_OF_AZKABAN_walk_f_cm_np1_ri_med_2
            #===
            #2

            imgPath = os.path.join(imageFolder, action, folder_to_use)
            posePath = os.path.join(poseFolder, action, folder_to_use)
            maskPath = os.path.join(maskFolder, action, folder_to_use)
            
            annot = scipy.io.loadmat(os.path.join(posePath, 'joint_positions'))
            bbox = scipy.io.loadmat(os.path.join(maskPath, 'Bbox.mat'))['Bbox']
            mask = scipy.io.loadmat(os.path.join(maskPath, 'puppet_mask.mat'))['part_mask']
            
            dicts = {'imgPath': imgPath, 'annot': annot, 'Bbox': bbox, 'mask': mask}
            
            if int(traintest) == 1:
                trainAnnot.append(dicts)
            else:
                testAnnot.append(dicts)

    return trainAnnot, testAnnot

class jhmdbDataset(data.Dataset):
    def __init__(self, trainAnnot, testAnnot, T, split):
        self.trainSet = trainAnnot[0:600]
        self.testSet = testAnnot
        self.valSet = trainAnnot[600:]
        self.inputLen = T
        self.split = split
        self.numJoint = 15
        self.min_scale = 0.8
        self.max_scale = 1.4
        self.max_degree = 40

        if self.split == 'train':
            self.dataLen = len(self.trainSet)
            self.p_scale = 1
            self.p_rotate = 1
            self.p_flip = 1
        if self.split == 'val':
            self.dataLen = len(self.valSet)
            self.p_scale = 0
            self.p_rotate = 0
            self.p_flip = 0
        else:
            self.dataLen = len(self.testSet)
            self.p_scale = 0
            self.p_rotate = 0
            self.p_flip = 0

        numData = len(self.trainSet)
        
        allSkeleton = []
        for i in range(0, numData):
            skeleton = self.trainSet[i]['annot']['pos_img']
            # (2, 15, 31)
            # (2, 15, 37)
            # (2, 15, 40)
            allSkeleton.append(skeleton)

        allSkeleton = np.concatenate((allSkeleton), 2)

        # print('1',self.dataLen)
        # print(allSkeleton.shape)(2, 15, 21066)

        self.meanX = np.expand_dims(np.mean(allSkeleton, axis=2)[0], 0)  # 1 x num_joint
        self.meanY = np.expand_dims(np.mean(allSkeleton, axis=2)[1], 0)
        self.stdX = np.expand_dims(np.std(allSkeleton, axis=2)[0],0)
        self.stdY = np.expand_dims(np.std(allSkeleton, axis=2)[1],0)

        # print('self.meanX',self.meanX.shape)(1, 15)
        # print('self.meanY',self.meanY.shape)(1, 15)
        # print('self.stdX', self.stdX)
        # print('self.stdY', self.stdY)

    def __len__(self):
        return self.dataLen
        # return 261               # TO DEBUG

    def read_annot(self, annotSet):
        imgPath = annotSet['imgPath']
        Bbox = annotSet['Bbox']
        # if test, replace 'skeleton' with baseline pose
        skeleton = annotSet['annot']['pos_img'].transpose(2, 1, 0)     # 2 x 15 x T ---> T x 15 x 2

        # /Users/makangbo/Desktop/Sub-JHMDB/Rename_Images/walk/EVOLUTION_walk_f_cm_np1_fr_med_23
        # (40, 4)
        # (40, 15, 2)

        return imgPath, Bbox, skeleton

    def preProcessImage(self, imgPath,skeleton):
        imgList = fnmatch.filter(os.listdir(imgPath), '*.png')
        imgList.sort()
        imgSequence = []
        random.seed(1234567890)
        scale_factor = 0
        rotate_angle = 0
        flip_factor = 0

        if random.random() < self.p_scale:
            scale_factor = random.uniform(self.min_scale, self.max_scale)
        if random.random() < self.p_rotate:
            rotate_angle = -self.max_degree + 2 * self.max_degree * random.random()
        if random.random() < self.p_flip:
            flip_factor = random.random()
        randoms = [scale_factor, rotate_angle, flip_factor]
        
        
        # print('!!',len(imgList))
        print(imgList)
        print('===')
        print(len(imgList))
        print('===')
        for i in xrange(len(imgList)):
            img_path = os.path.join(imgPath, imgList[i])
            input_image = Image.open(img_path)
            
            countcenter = np.array(input_image).shape
            center_x = int(countcenter[0]/2)
            center_y = int(countcenter[1]/2)
            center   = [center_x, center_y]

            #img = np.array(input_image).astype('uint8')
            # print(img.shape)(240, 320, 3)\
            if i == 10:
                img = np.array(input_image).astype('uint8')
                print('img.shape',img.shape)
                np.save('/Users/makangbo/Desktop/ceshiprpic.npy',img)
            #可视化
            # img =  np.array(input_image).astype('uint8')
            # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  #
            # cv2.imshow('a',img)
            # cv2.waitKey(500)
            
            # based on official setup from Resnet
            
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            preprocess2 = Mytransforms_JHMDB.Compose([
                        Mytransforms_JHMDB.RandomResized(),
                        Mytransforms_JHMDB.RandomRotate(40),
                        Mytransforms_JHMDB.SinglePersonCrop(256),
                        Mytransforms_JHMDB.RandomHorizontalFlip(),         
                    ])
            

            if i == 10:
                print('skeleton[10,:,:].shape',skeleton[10,:,:].shape)
                print('center',center)
                print('random',randoms)
                xxx,skeleton = preprocess2(np.array(input_image),skeleton[10,:,:],center,randoms)
                print(np.array(xxx).shape)
                print(i)
                np.save('/Users/makangbo/Desktop/ceshipic.npy',xxx)
                np.save('/Users/makangbo/Desktop/ceshiskele.npy',skeleton)
            
            img_tensor = preprocess(input_image)

            imgSequence.append(img_tensor.unsqueeze(0))

        imgSequence = torch.cat((imgSequence), 0)
        # print('xxx',np.array(imgSequence).shape)(40, 3, 224, 224)

        return imgSequence,skeleton

    def data_to_use(self, skeleton, bbox, imgSequence):

        nframes = skeleton.shape[0]
        random.seed(1234567890)
        useLen = self.inputLen

        #print('skeleton',skeleton.shape)
        #print('bbox',bbox.shape)
        #print('imgSequence',imgSequence.shape)
        #print('uselen',useLen)

        # skeleton(35, 15, 2)
        # bbox (35, 4)
        # imgSequence  torch.Size([35, 3, 224, 224])
        # uselen 40

        if len(skeleton.shape) == 4:
            skeleton = skeleton.squeeze(0)  # inputLen x 15 x 2
            bbox = bbox.squeeze(0)  # T x 4
            imgSequence = imgSequence.squeeze(0)
            # baselineSkeleton = baselineSkeleton.squeeze(0)
            'case 1'
        if nframes > useLen:
            'if training, idx is randomly sampled; if testing, starting from first frame'
            idx = random.randint(0, nframes - useLen) # training and val
            # idx = 0  # testing, rest of frames belongs to 'case 3'

            data_sel = np.expand_dims(skeleton[idx: idx + useLen], 0)
            bbx_sel = np.expand_dims(bbox[idx: idx + useLen], 0)
            img_sel = np.expand_dims(imgSequence[idx: idx + useLen], 0)
            sequence_to_use = data_sel
            bbox_to_use = bbx_sel
            imgSequence_to_use = img_sel
            mask_idx = np.ones((sequence_to_use.shape))


            'case 2'
        elif nframes == useLen:
            idx = 0
            sequence_to_use = skeleton
            # baseline_to_use = baselineSkeleton
            mask_idx = np.ones((sequence_to_use.shape))

     
            if bbox.shape[0] == nframes:
                bbox_to_use = bbox
                imgSequence_to_use = imgSequence

            else:
                bbox_to_use = bbox[0:nframes]   # need to check'
                imgSequence_to_use = imgSequence[0:nframes]

            'case 3'
        else:
            seqLeft = useLen - nframes
            sequence = []
            # base = []
            img_sequence = []
            bbx = []
            m_idx = []
            idx = 0   # start from first frame

            for i in xrange(seqLeft):
                # data_sel = skeleton[-1]   # last frame
                #
                # bbx_sel = bbox[-1,:]
                mask_sel = np.zeros((self.numJoint, 2))
                data_sel = np.zeros((self.numJoint, 2))
                # base_sel = np.zeros((self.numJoint, 2))   # baseline joint is 14
                bbx_sel = np.zeros((4))
                img_sel = torch.zeros(3, 224, 224)

                sequence.append(np.expand_dims(data_sel, 0))
                # base.append(np.expand_dims(base_sel, 0))
                bbx.append(np.expand_dims(bbx_sel, 0))
                img_sequence.append(img_sel.unsqueeze(0))
                m_idx.append(np.expand_dims(mask_sel, 0))

                # print('1',np.array(sequence).shape)
                # print('2',np.array(bbx).shape)
                #print('2',np.array((img_sel.unsqueeze(0)).shape))
                #print('3',np.array(img_sequence).shape)

                # (1, 1, 15, 2)
                # (1, 1, 4)
                #[  1   3 224 224]

            sequence = np.concatenate(sequence, axis=0)   # seqLeft x 15 x 2
            bbx = np.concatenate(bbx, axis=0)  # seqLeft x 4
            sequence_img = torch.cat((img_sequence), 0)
            ma_idx = np.concatenate(m_idx, axis=0)
            
            #print('1',sequence.shape)
            #print('2',sequence_img.shape)
            
            # base = np.concatenate(base, axis=0)

            # sequence_to_use = torch.Tensor(np.concatenate((skeleton.numpy(), sequence), axis=0)).type(torch.FloatTensor).unsqueeze(0)  # 1 x inputLen x 15 x 2
            # bbox_to_use = torch.Tensor(np.concatenate((bbox.numpy(), bbx), axis=0)).type(torch.FloatTensor).unsqueeze(0)

            sequence_to_use = np.concatenate((skeleton, sequence), axis=0).astype(float)
            # baseline_to_use = np.concatenate((baselineSkeleton, base), axis=0).astype(float)
            mask_part1 = np.ones((skeleton.shape))
            mask_idx = np.concatenate((mask_part1, ma_idx), axis=0).astype(float)
            
            if bbox.shape[0] == nframes:      
                bbox_to_use = np.concatenate((bbox, bbx), axis=0).astype(float)
                imgSequence_to_use = torch.cat((imgSequence.float(), sequence_img.float()), 0).type(torch.FloatTensor)
            else:
                bbox_to_use = np.concatenate((bbox[0:nframes], bbx), axis=0).astype(float)
                imgSequence_to_use = torch.cat((imgSequence[0:nframes], sequence_img), 0).type(torch.FloatTensor)

        return sequence_to_use, bbox_to_use, imgSequence_to_use, mask_idx, nframes, idx

    def get_normalized_data(self, skeleton):
        'skeleton : 15 x 2 x T'
        
        # skeleton = torch.Tensor(annotSet['annot']['pos_img']).permute(2, 1, 0).type(torch.FloatTensor)
        # nframes = skeleton.shape[2]
        # skeleton = sequence_to_use.squeeze(0)

        X = skeleton[:,:,0]
        Y = skeleton[:,:,1]

        # meanX_mat = torch.FloatTensor(self.meanX).repeat(1, nframes)
        # meanY_mat = torch.FloatTensor(self.meanY).repeat(1, nframes)
        # stdX_mat = torch.FloatTensor(self.stdX).repeat(1, nframes)
        # stdY_mat = torch.FloatTensor(self.stdY).repeat(1, nframes)
        # normX = (X - meanX_mat)/stdX_mat
        # normY = (Y - meanY_mat)/stdY_mat
        #
        # normSkeleton = torch.cat((normX.unsqueeze(0), normY.unsqueeze(0)), 0).type(torch.FloatTensor)

        normX = (X - self.meanX)/self.stdX
        normY = (Y - self.meanY)/self.stdY
        
        normSkeleton = np.concatenate((np.expand_dims(normX,2), np.expand_dims(normY,2)), 2).astype(float) # inputLen x 15 x 2
        
        return normSkeleton

    def get_unNormalized_data(self, normSkeleton):
        'for inference part, normSkeleton : N x useLen x 15 x 2'

        # nframes = normSkeleton.shape[2]
        if len(normSkeleton.shape) == 4:
            normSkeleton = normSkeleton.squeeze(0)

        meanX_mat = torch.FloatTensor(self.meanX).repeat(self.inputLen, 1)   # inputLen x 15
        meanY_mat = torch.FloatTensor(self.meanY).repeat(self.inputLen, 1)
        stdX_mat = torch.FloatTensor(self.stdX).repeat(self.inputLen, 1)
        stdY_mat = torch.FloatTensor(self.stdY).repeat(self.inputLen, 1)

        X = normSkeleton[:,:,0]  # inputLen x 15
        Y = normSkeleton[:,:,1]  # inputLen x 15

        unNormX = X * stdX_mat + meanX_mat
        unNormY = Y * stdY_mat + meanY_mat

        unNormSkeleton = torch.cat((unNormX.unsqueeze(2), unNormY.unsqueeze(2)), 2)
        
        return unNormSkeleton
        
    def __getitem__(self, idx):
        if self.split == 'train':
            annotSet = self.trainSet[idx]
        else:
            annotSet = self.testSet[idx]

        imgPath, Bbox, Skeleton = self.read_annot(annotSet)
        

        # imgPath, Bbox, Skeleton, baseline = self.read_annot(annotSet)

        # normSkeleton = self.get_normalized_data(Skeleton)
        # imgSequence = self.preProcessImage(imgPath, Bbox)
        imgSequence,Skeleton = self.preProcessImage(imgPath,Skeleton)
        
        print(imgPath)
        print('===')
        print(Bbox.shape)
        print('===')
        print(Skeleton.shape)
        
        np.save('/Users/makangbo/Desktop/ceshibbox.npy',Bbox[10,:])
        np.save('/Users/makangbo/Desktop/ceshiskeleton.npy',skeleton[10,:,:])
        
        # baseline_to_use = self.get_normalized_data(baseline)

        # print('Skeleton', Skeleton.shape)
        # print('Bbox', Bbox.shape)
        # print('imgSequence', imgSequence.shape)

        # imgSequence
        # torch.Size([40, 3, 224, 224])
        # Skeleton(40, 15, 2)
        # Bbox(40, 4)
        
        sequence_to_use, Bbox_to_use, imgSequence_to_use, mask_idx, nframes, idx = self.data_to_use(Skeleton, Bbox, imgSequence)
               
        dicts = {'imgSequence_to_use': imgSequence_to_use, 'Bbox_to_use': Bbox,
                 'baseline_to_use':sequence_to_use, 'nframes':nframes,'imgPath':imgPath}
        
        '''
        dicts = {'imgSequence_to_use': imgSequence_to_use, 'Bbox_to_use': Bbox_to_use,
         'sequence_to_use': sequence_to_use,'baseline_to_use':baseline, 'mask_idx': mask_idx, 'nframes':nframes,
         'randomInd:':idx, 'imgPath':imgPath}    
        '''
        
        return dicts

if __name__ =='__main__':
    dataRoot = '/Users/makangbo/Desktop/Sub-JHMDB'
    
    trainAnnot, testAnnot = get_train_test_annotation(dataRoot)
    
    N = np.zeros(len(testAnnot))
    for i in range(0, len(testAnnot)):
        nframes = testAnnot[i]['annot']['pos_img'].shape[2]
        N[i] = (nframes)
    
    dataset = jhmdbDataset(trainAnnot, testAnnot, T=40, split='test')
    
    for i,sample in enumerate(dataset):
        x = sample['imgSequence_to_use']
        y = sample['baseline_to_use']
           
    
    '''
    dloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    
    for i, sample in enumerate(dloader):
        
        # sequence_to_use = sample['sequence_to_use']
        
        baseline = sample['baseline_to_use']
        Bbox = sample['Bbox_to_use']
        imgSequence_to_use = sample['imgSequence_to_use']
        # mask_idx = sample['mask_idx']
        
        print('sample:', i, 'squence shape:', baseline.shape,
              'bbox shape:' , Bbox.shape, 'imgSeq:', imgSequence_to_use.shape)
        
                
        #ss shape: torch.Size([1, 40, 15, 2])
        #bbox: torch.Size([1, 40, 4])
        #imgSeq: torch.Size([1, 40, 3, 224, 224])
        #mask_idx: torch.Size([1, 40, 15, 2])
        
        
        np.save('/Users/makangbo/Desktop/sequence_to_use',np.array(baseline))
        np.save('/Users/makangbo/Desktop/Bbox',np.array(Bbox))
        np.save('/Users/makangbo/Desktop/imgSequence_to_use',np.array(imgSequence_to_use))
        
        # if i == 0:
        #     break
        
    '''
