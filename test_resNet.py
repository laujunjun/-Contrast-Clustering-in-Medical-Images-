import numpy as np
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from models.dataset import *

#import resnet
from models import resnet, model
#from models import resnet
#from model_Start import *
import math
import torch.cuda.amp as amp
import torchvision
import argparse
from utils import yaml_config_hook, save_model
#from ctran import ctranspath #addnew
#import torch.nn as nn #addnew
import ResNet as ResNet #addnew
def image_to_tensor(image, mode='bgr'):  # image mode
    if mode == 'bgr':
        image = image[:, :, ::-1]
    x = image
    x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    x=torch.unsqueeze(x, dim=0)
    return x
def Path_to_Image(Path,image_size=224):
    image = cv2.imread(Path)
    # b, g, r = cv2.split(image)
    # image = cv2.merge([r, g, b])
    image = image.astype(np.float32) / 255
    image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image=image_to_tensor(image)
    return image


def Get_Evaluation_Metrics(output,batch):
    Weight=np.array(output['instance_projector_i'].detach().cpu())

    y_Pre = np.array(output['Cluste'].detach().cpu())
    y_name=np.array(batch['name'])

    return y_Pre,Weight,y_name
def initialization():
    Y_Sample_Name_List=np.array(0,dtype=np.str_)
    Y_Weight_List=np.zeros((1,128),dtype=np.float32)

    Y_Pre_List = np.array([0]).reshape(-1, 1)
    Y_True_List = np.array([0]).reshape(-1, 1)

    return Y_Sample_Name_List,Y_Weight_List,Y_Pre_List,Y_True_List

if __name__ == '__main__':
    res = resnet.get_resnet("ResNet50")

    parser = argparse.ArgumentParser()
    config = yaml_config_hook("/home/ljj/code/Pathosig_master/code/1_deeplearning_contrastive_cluster/config/config_resnet.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    out_dir = args.out_dir
    Model_Name=['/path/00000000.models.pth'] # 20240805 TCGA
    for name in Model_Name:
        Model_Path=name
        ID=name.split('.')[0]
        weights_path = Model_Path
        batch={}

        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.Net(arg=args,resnet=res).to(device) 
        # model.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage)['state_dict'], strict=False)
        model.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage), strict=False)
        model=model.to(device)
        model=model.eval()
        
        # # 模型加载时缺少“state_dict”键值，故直接使用加载的字典对象，而不是尝试访问 'state_dict' 键
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = model.Net(arg=args,resnet=res).to(device) 
        # model_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
        # model.load_state_dict(model_dict, strict=False)
        # model=model.to(device)
        # model=model.eval()


        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #addnew
        #res = ctranspath() #addnew
        #res.head = nn.Identity() #addnew
        #model = model.Net(arg=args, resnet=res).to(device) #addnew
        #model=model.eval()

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #res = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        #res.fc = nn.Identity()
        #model = model.Net(arg=args, resnet=res).to(device) #addnew
        #model=model.eval()

        Val_1_type=pd.read_csv('/all_patch_label.csv',sep=',')
        
        #Val_2_type = pd.read_csv("Val_2.csv", sep=',')
        test_dataset = HubmapDataset(Val_1_type, args, valid_augment5, valid_augment5)

        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=200,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=null_collate,
        )
        Test_Name_List, Test_Weight_List, Test_Pre_List, Test_True_List = initialization()
        for t, batch in enumerate(test_loader):
            a=1
            model.output_type = ['loss', 'inference']
            with torch.no_grad():
                batch_size = len(batch['index'])
                batch['image'] = batch['image'].cuda()
                batch['image_Argument'] = batch['image_Argument'].cuda()
                output = model(batch)

                PreDice=output['Cluste'].cpu().detach().numpy()
                y_Pre,Weight, y_name = Get_Evaluation_Metrics(output,batch)
            Test_Pre_List = np.hstack((Test_Pre_List, np.array(y_Pre).reshape(1,-1)))
            Test_Name_List = np.hstack((Test_Name_List, y_name))
            Test_Weight_List=np.vstack((Test_Weight_List,Weight))
        My_Train_PD = pd.DataFrame({'Name': Test_Name_List[1:],
                                    'Pre_Label': Test_Pre_List[0][1:]})
        Index_cluster = ['Cluster:{}'.format(i) for i in range(128)]
        Index = np.hstack((np.array(["Name"]), np.array(Index_cluster)))
        Values=np.hstack((np.array(Test_Name_List[1:]).reshape(-1,1),Test_Weight_List[1:,:]))
        My_Weight=pd.DataFrame(Values,columns=Index)
        MyFile=pd.merge(My_Weight,My_Train_PD,how='inner',on='Name')
        #MyFile.to_csv('Val_1_type_{}_Umap.csv'.format(ID))
        # 创建目录，如果目录不存在
        os.makedirs(ID, exist_ok=True)
        MyFile.to_csv('/path/feature_128_allData_epoch0_trainedBy10Slides.csv'.format(ID))

