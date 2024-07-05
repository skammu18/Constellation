## ConstellationAI
This project is set out to classify different constellations based on four major categories: 'aries' 'libra' 'pisces' and 'virgo.' As someone who always stared up at the sky and thought everything looked the same, I thought this project could help a lot of people learn about how much really exists up there. This project can help anyone who wants to identify different constellations and learn more about the sky.

This image is of the libra constellation, and it is correctly classified as libra:(https://github.com/skammu18/Constellations/assets/173948115/33873b03-73c1-4790-82d5-7e3563cde31f)

## The Algorithm

This project classifies constellation images based on four categories: aries, libra, pisces, and virgo. The model is first trained using a dataset, where it identifies key features from the images to be able to later classify new images. The model is trained through the already existing resnet18 network, but is being retrained with new constellation images. The model is then validated, exported, and tested. The user can run the python script to select an image for the model to classify. 

## Running this project

1. Collect a dataset of images for each aries, libra, pisces, and virgo constellations. You can also use Kaggle to download a dataset of images. [Dataset I used for images: https://www.kaggle.com/datasets/tejask257/star-constellation] 
2. Within the dataset create three folders titled 'test' 'train' and 'val' . Within each of these folders create four new separate folders titled 'aries' 'libra' 'pisces' and 'virgo'. Put in your corresponding constellation images into each of these folders.
3. Give your dataset a name, such as constellations, and upload it to VS Code by dragging and dropping.
4. Within your dataset file create another file titled 'labels.txt'. In this file type aries, libra, pisces, and virgo on separate lines.
5. Go to nvidia/jetson-inference/ directory by entering: cd nvidia/jetson-inference/
6. Perform a memory command to ensure your nano has enough space to train: 'echo 1 | sudo tee /proc/sys/vm/overcommit_memory'
7. cd to your jetson interfence folder. Then enter the docker container by performing: './docker/run.sh' [enter password when prompted]
8. Change directories to jetson-inference/python/training/classification
9. Train your model: 'python3 train.py --model-dir=models/[your file name] data/[your file name]'
10. Make sure you are still in the docker container and within the jetson-inference/python/training/classification directory.
11. Run the export script: python3 onnx_export.py --model-dir=models/[your file name]>
12. Look in the jetson-inference/python/training/classification/models/cat_dog folder and ensure a file titled 'resnet18.onnx' exists. This is you retrained model.
13. Exit the docker container by typing 'exit' or 'ctrl +d'
14. Type and enter 'ls models/[your filename]/' . Ensure the resnet18.onnx file is there.
15. Set the NET variable: NET=models/[your file nam]>
16. Set the DATASET variable: DATASET=data/[your file name]>
17. Test a single image: imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/[chosen category]/[chosen testing image name].jpg result.jpg
18. Click on your image to see what constellation your model classified the test image as and what corresponding percentage it gives.



VIDEO DEMO:
https://github.com/skammu18/Constellations/assets/173948115/e59b0735-1696-4a76-8ac1-fdc365ab85d1 


## Running the Program Through Python

1. Create a python image classification program in order to effectively run your model each time you check an image.
2. First, create a new file under your project folder titled constellations.py
3. To automatically access the python interpreter add this code: #!/usr/bin/python3
4. Import the jetson modules required to load and process the images:
   
import jetson_inference

import jetson_utils

import argparse

5. To parse the image file:
   
   parser = argparse.ArgumentParser()

   parser.add_argument("filename", type=str, help="filename of the image to process")

6. Add: opt = parser.parse_args()

7. Add the code that allows you to load an image: img = jetson_utils.loadImage(opt.filename)

8. Add: net = jetson_inference.imageNet(model="resnet18.onnx",labels="labels.txt",input_blob="input_0", output_blob="output_0")

9. Classify the image: class_idx, confidence = net.Classify(img)

10. Get the class description: class_desc = net.GetClassDesc(class_idx)

11. Output result: print("This constellation is "+ str(class_desc) +" (class #"+ str(class_idx) +") with " + str(confidence*100)+"% confidence")

12. To download an image, return to your terminal and type ‘wget [link to image]’

13. To view the final output and classification, type: python3 constellations.py [name of image file].jpg



