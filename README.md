# B-CellClassification
Classifying malignant cells in stained microscopic white blood cell images with high accuracy.

Texture provides significant information for classification in this problem hence we are using bilinear features from a trained CNN model. After the general classification model, we appended [bilinear pooling](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf) to compute the outer product of the deep descriptors.

Pretrained Classification models tried on the dataset - AlexNet, VGG16

![Cancer cell images](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)


![Normal healthy cell images](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)


Append [Stain deconvolution layer](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_50) before AlexNet/VGG(In Process)

Data Augmentation-
1. Random Crop
2. Random Horizontal Flip
3. Random Vertical Flip
4. Random Resize

Make sure to keep cell_dataset according to the following structure:

B-CellClassification  
|  
|------doc  
|------src  
|------cell_dataset  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-----all  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-----hem  
        
Validation Accuracy - 83%  
Test Accuracy - 77%  
To fine-tune FC layer Run the command-  
python ./src/bilinear_cnn_fc.py --base_lr 1.0 --batch_size 64 --epochs 55 --weight_decay 1e-8  

To fine-tune all layers Run the command-  
python ./src/bilinear_cnn_all.py --base_lr 1e-2 --batch_size 64 --epochs 25 --weight_decay 1e-5 --model "model.pth"  

Make sure "model.pth" points to the proper model which was fine tuned before  

Versions used:  
Anaconda (4.0.0)  
Python (3.5.1)  
Pytorch (0.4.1)  
NVIDIA-SMI (384.130)  
CUDA - (release 9.0, V9.0.176)  


