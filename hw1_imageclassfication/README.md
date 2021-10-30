# hw1_imageclassfication


## Hardware
● Windows 10 <br>
● Intel(R) Core i7-7700HQ CPU  <br>
● NVIDIA GeForce GTX 1050 Ti <br>

## Introduction and details
There is the outlines in this competition <br>
1. [Installation](#Installation) <br>
2. [Dataloader](#Dataloader) <br>
3. [Model](#Model) <br>
4. [Load-Model](#Load-Model) <br>
5. [testing](#testing) <br>
6. [Make-Submission](#Make-Submission) <br>

## Installation
Using Anaconda and pytorch to implement this method.

    conda create -n Classification python=3.6
    conda install pytorch -c pytorch
    conda install torchvision -c pytorch

## Dataloader
Change the path which is in the `hw1.py`.
    
    train_dir = './2021VRDL_HW1_datasets/training_images'
    with open('./2021VRDL_HW1_datasets/training_labels.txt', 'r', encoding='utf-8-sig') as R:
training and testing data are ues the same method to load <br>
the valid_set will be split in by train_set in 8:2<br>

    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size])
    

## Model
Taking the pretrained model for resnet50 <br>
    
    net =models.resnet50( pretrained=pretrained).to(device)


### Data_transforms

     'train': transforms.Compose([
        transforms.Resize(256),
           torchvision.transforms.RandomOrder([
           torchvision.transforms.RandomCrop((256, 256)),
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.RandomVerticalFlip()
        ]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        torchvision.transforms.CenterCrop((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
I put the AutoAugment to transforms the data which is `newImageTransform()` <br>


### Optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)
   
It let the accuracy increase a lot, when my model is stuck. <br>
 
## Training 
Taking training in `train_model()`<br>
    
                        
## Load-Model
After processing every epoch, you need to save the model, in oder to avoid the model breaking. <br>   
You can load model as the code： 

    checkpoint = net.load_state_dict(torch.load('./best_checkpoint_last.pth'))
   

## testing
Using this code to do the testing_img_order. <br>

    with open('./2021VRDL_HW1_datasets/testing_img_order.txt') as f:
    
## Make-Submission
Submit the file `answer.txt`, to the file <br>
