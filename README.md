## ConstellationAI
This project is set out to classify different constellations based on four major categories: 'aries' 'libra' 'pisces' and 'virgo.' As someone who always stared up at the sky and though everything looked the same, I though this project could help a lot of people learn about how much really exists up there. This project can help anyone who wants to identify different constellations and learn more about the sky.

This image is of the libra constellation, and it is correctly classified as libra:(https://github.com/skammu18/Constellations/assets/173948115/33873b03-73c1-4790-82d5-7e3563cde31f)

## The Algorithm

This project classifies constellation images based on four categories: aries, libra, pisces, and virgo. The model is first trained using a dataset, where it identifies key features from the dataset of images to be able to later classify new images. The model is trained through the already existing resnet18 network, but is being retrained with new constellation images. The model is then validated, exported, and tested. The user can run the python script to select an image for the model to classify. 

## Running this project

1. Collect a dataset of images for each aries, libra, pisces, and virgo constellations. You can also use Kaggle to download a dataset of images.
2. Within the dataset create three folders titled 'test' 'train' and 'val' . Within each of these folders create four new separate folders titled 'aries' 'libra' 'pisces' and 'virgo'. Put in your corresponding constellation images into each of these folders.
3. Give your dataset a name, such as constellations, and upload it to VS Code by dragging and dropping.
4. Within your dataset file create another file titled 'labels.txt'. In this file type aries, libra, pisces, and virgo on sepearate lines.
5. Go to nvidia/jetson-inference/ directory by entering: cd nvidia/jetson-inference/
6. Perform a memory command to ensure your nano has enough space to train: 'echo 1 | sudo tee /proc/sys/vm/overcommit_memory'
7. cd to your jetson interfence folder. Then enter the docker container by performing: './docker/run.sh' [enter password when prompted]
8. Change directories to jetson-inference/python/training/classification
9. Train your model: 'python3 train.py --model-dir=models/<your file name> data/<your file name>'
10. Make sure you are still in the docker container and within the jetson-inference/python/training/classification directory.
11. Run the export script: python3 onnx_export.py --model-dir=models/<your file name>
12. Look in the jetson-inference/python/training/classification/models/cat_dog folder and ensure a file titled 'resnet18.onnx' exists. This is you retrained model.
13. Exit the docker container by typing 'exit' or 'ctrl +d'
14. Type and enter 'ls models/<your file name>/' . Ensure the resnet18.onnx file is there.
15. Set the NET variable: NET=models/<your file name>
16. Set the DATASET variable: DATASET=data/<your file name>
17. Test a single image: imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/<chosen category>/<chosen testing image name>.jpg result.jpg
18. Click on your image to see what constellation your model classified the test image as and what corresponding percentage it gives.



VIDEO DEMO:
https://github.com/skammu18/Constellations/assets/173948115/e59b0735-1696-4a76-8ac1-fdc365ab85d1 

