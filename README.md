# Image Unfogging
A neural network writen in PyTorch to unfog an image or stbilize the colour balance by unhazing.

## Architecture
### AOD-Net: 
The Project tries to replicate the architecture outlined in the project [AOD-Net](https://sites.google.com/site/boyilics/website-builder/project-page). <br/>
AOD-Net stands for All in One Dehazing Network which is a lightweight image dehazing network that is fast to train and using for inference.

## Training
**Preparation:**
1. Create folder "data".
2. Create 2 subfolders, "images" and "data" with original images and foggy images respectively.
3. Install the requirements of the project by typing ```sudo pip3 install -r requirements.txt```

**Training:**
1. To traning the model run: 
```{python}
python3 train.py
```
2. Open train.py file for details of the parameters and parser arguments.

3. Random validation samples will be saved to "samples" directory after every epoch and the model will be saved as "net.pth" in the "snapshots" directory. A pretrained model has been provided in the snapshots directory.

## Testing/Running the pretrained model:
1. Copy the images to be unfogged in the "test_images" directory and run 
```{python}
python3 main.py
```
<br/>
The net will save the unfogged images in the "results" directory.

**Results:**

The models performs well in removing the fog and haziness in the foreground of the input images. for removing complete fog it would require some sort of Generative nets to replace the foggy patches where nothing is seen.

Some of the results are:

![result](results/canyon.png?raw=true "Title")  
![result](results/test11.jpg?raw=true "Title")  
![result](results/test2.jpg?raw=true "Title")  
![result](results/test6.jpg?raw=true "Title")  
![result](results/test14.jpg?raw=true "Title")  
![result](results/test.png?raw=true "Title")
