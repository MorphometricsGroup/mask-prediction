import os
from stqdm import stqdm
import cv2
import torch
from torch.utils.data import Dataset as BaseDataset
import albumentations as A
import segmentation_models_pytorch as smp

# Definition of dataset
class Dataset(BaseDataset):
    def __init__(self, data_dir, augmentation=None, preprocessing=None):
        self.image_paths = data_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentation is not None:
            sample = self.augmentation(image=image)
            image = sample['image']
        
        transform = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)
        sample = transform(image=image)
        image = sample['image']
        
        if self.preprocessing is not None:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        image = torch.from_numpy(image) 
        
        return image


# Preparation of model
def prepare_model(encoder, encoder_weights, classes, activation, device, model_parameters_path):
    # Define model
    model = smp.Unet(
    encoder_name=encoder,
    encoder_weights=encoder_weights,
    in_channels=3,
    classes=len(classes),
    activation=activation
    )
    
    model.to(device)
    
    # Install trained parameters in model
    model.load_state_dict(torch.load(model_parameters_path, map_location=device)) # cpu or cuda
    model.eval()
    
    return model

# Creation of dataset
def create_dataset(data_dir, encoder, encoder_weights):
    # Define augmentation for inference
    def get_inference_augmentation():
        transform = A.Compose([
            A.Resize(width=512, height=512, p=1.0)
        ], p=1.0)
        return transform

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32') 

    def get_preprocessing(preprocessing_fn):
        _transform = A.Compose([
            A.Lambda(image=preprocessing_fn, p=1.0),
            A.Lambda(image=to_tensor, mask=to_tensor, p=1.0)
        ], p=1.0)
        return _transform
            
    # Set pretrained parameters
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    
    # Create dataset
    dataset = Dataset(
        data_dir=data_dir, 
        augmentation=get_inference_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    return dataset

# Inference    
def inference(model, dataset, data_dir, classes, device, mask_folder_path):
    def one_hot_function(pr_mask):
        max_index_number = pr_mask.argmax(axis=0) 
        for i in range(len(pr_mask)):
            one_hot_vector = (max_index_number == i) 
            pr_mask[i] = one_hot_vector
        return pr_mask
    
    for i in stqdm(range(len(dataset)), ncols=80):
        image = dataset[i]
        x_tensor = image.to(device).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy()
        pr_mask = one_hot_function(pr_mask)
        pr_mask = pr_mask[classes.index('soybean')] # Extract soybean
        image_path = data_dir[i]
        assert dataset.image_paths[i] == image_path, "Not match pathes"
        raw_image = cv2.imread(image_path)
        height, width = raw_image.shape[:2]
        pr_mask = A.resize(pr_mask, height, width)
        pr_mask = pr_mask * 255
        pr_mask_path = os.path.normpath(os.path.join(mask_folder_path, os.path.basename(data_dir[i]).split('.')[0]+'.png')) # Save masks as PNG format
        cv2.imwrite(pr_mask_path, pr_mask)
        os.remove(image_path) # Delete image file

def inference_2(model, dataset, data_dir, classes, device, mask_folder_path):
    def one_hot_function(pr_mask):
        max_index_number = pr_mask.argmax(axis=0) 
        for i in range(len(pr_mask)):
            one_hot_vector = (max_index_number == i) 
            pr_mask[i] = one_hot_vector
        return pr_mask
    
    for i in stqdm(range(len(dataset)), ncols=80):
        image = dataset[i]
        x_tensor = image.to(device).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy()
        pr_mask = one_hot_function(pr_mask)
        pr_mask_1 = pr_mask[classes.index('soybean')] # Extract soybean
        pr_mask_2 = pr_mask[classes.index('stage')] # Extract stage
        pr_mask = pr_mask_1 + pr_mask_2
        image_path = data_dir[i]
        assert dataset.image_paths[i] == image_path, "Not match pathes"
        raw_image = cv2.imread(image_path)
        height, width = raw_image.shape[:2]
        pr_mask = A.resize(pr_mask, height, width)
        pr_mask = pr_mask * 255
        pr_mask_path = os.path.normpath(os.path.join(mask_folder_path, os.path.basename(data_dir[i]).split('.')[0]+'.png')) # Save masks as PNG format
        cv2.imwrite(pr_mask_path, pr_mask)
        os.remove(image_path) # Delete image file