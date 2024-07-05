#!/usr/bin/python3
import jetson_inference
import jetson_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
opt = parser.parse_args()
img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(model="resnet18.onnx",labels="labels.txt",input_blob="input_0", output_blob="output_0")

class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)
print("This constellation is "+ str(class_desc) +" (class #"+ str(class_idx) +") with " + str(confidence*100)+"% confidence")

