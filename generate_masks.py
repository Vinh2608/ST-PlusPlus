import torch
from torchvision import transforms
from PIL import Image
from dataset.semi import SemiDataset
import numpy as np
import matplotlib.pyplot as plt
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from model.semseg.unet import UNet
from utils import count_params, meanIOU, color_map
import tqdm
import os
from torch.utils.data import DataLoader


# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)), # Resize to match the input size expected by the model
    transforms.ToTensor(),         # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with ImageNet stats
])
def init_basic_elems(name, args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus,
                 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    # This is old code
    # model = model_zoo[args.model](args.backbone, 21 if args.dataset == 'pascal' else 19)

    # This is for dataset1 and dataset2
    if name != 'unet':
        model = model_zoo[name]('resnet101', 3)

    else:
        model = UNet('resnet101', 3, args)

    return model
# Function to load the model from checkpoint
def load_model(checkpoint_path):
    model = UNet('resnet101', 3, 'dataset2')
    #model = torch.nn.DataParallel(model)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.to('cuda:0')
    return model

def label(model, dataloader, args):
    model.eval()
    #tbar = tqdm(dataloader)

    global MODE
    # This is old code
    # metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)

    cmap = color_map('dataset2')

    with torch.no_grad():
        for img, mask, id in dataloader:
            img = img.cuda()
            
            pred = model(img, True)

            pred = torch.argmax(pred, dim=1).cpu()
            # # Create a new figure
            # plt.figure(figsize=(10,10))

            # plt.subplot(1, 3, 1)
            # plt.imshow(img.cpu().detach().numpy().squeeze(0).transpose(1,2,0))
            # plt.title('Image')

            # # Plot `mask` on the left
            # plt.subplot(1, 3, 2)
            # plt.imshow(mask.numpy().squeeze(0), cmap='gray')
            # plt.title('Mask')

            # # Plot `pred` on the right
            # plt.subplot(1, 3, 3)
            # plt.imshow(pred.numpy().squeeze(0), cmap='gray')
            # plt.title('Prediction')

            # # Display the figure
            # plt.show()

            pred = Image.fromarray(pred.squeeze(
                0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)
            pred.save('%s/%s' % ('dataset2MaskNormal',
                          os.path.basename(id[0].split(' ')[1])))
# Function to preprocess the image and generate the segmentation mask
def generate_segmentation_mask(model, image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Create a mini-batch as expected by the model
    
    # Move input tensor to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)
    
    # Generate the prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-process the output to get the segmentation mask
    # Assuming the model outputs logits, apply softmax and get the class with the highest probability
    output = torch.softmax(output, dim=1)
    output_mask = output.argmax(dim=1).cpu().numpy()[0]
    
    return output_mask

# Function to visualize the segmentation mask
def visualize_segmentation(image_path, segmentation_mask):
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_mask, cmap='jet')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.show()

# Main function
if __name__ == "__main__":
    #checkpoint_path = 'outdir/models/lisc/1_4/split_random/unet/resnet101/circular_cutout/self_training++/non_consistency_training/first_time/unet_resnet101_76.08.pth'
    #checkpoint_path = 'outdir/models/lisc/1_4/split_random/unet/resnet101/self_training++/non_consistency_training/first_time/unet_resnet101_76.30.pth'
    #checkpoint_path = 'outdir/models/dataset2/1_4/split_random/unet/resnet101/circular_cutout/self_training++/non_consistency_training/first_time/unet_resnet101_88.28.pth'
    checkpoint_path = 'outdir/models/dataset2/1_4/split_random/unet/resnet101/self_training++/non_consistency_training/second_time/unet_resnet101_88.91.pth'
    valset = SemiDataset('dataset2', './', 'val2', None)
    valloader = DataLoader(valset, batch_size=1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    
    # Load the model
    model = load_model(checkpoint_path)
    label(model,valloader, None)
    
    # Generate the segmentation mask
    # segmentation_mask = generate_segmentation_mask(model, image_path)
    
    # # Visualize the segmentation mask
    # #for img in os.listdir(image_path):
    # visualize_segmentation(image_path, segmentation_mask)
