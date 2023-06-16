# This is a sample Python script.
import torch
import torchvision
from imageai.Detection import ObjectDetection

# instantiating the class  
recognizer = ObjectDetection()  

# defining the paths  
path_model = "models"
path_input = "input/mid_journey_2.jpg"
path_output = "output/test.jpg"



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    recognizer = ObjectDetection()

    # defining the paths
    path_model = "models/yolov3.pt"



    recognizer.setModelTypeAsYOLOv3()
    recognizer.setModelPath(path_model)

    recognizer.loadModel()

    # calling the detectObjectsFromImage() function
    recognition = recognizer.detectObjectsFromImage(
        input_image=path_input,
        output_image_path=path_output,
        minimum_percentage_probability = 10
    )

    # iterating through the items found in the image
    for eachItem in recognition:
        print(eachItem["name"], " : ", eachItem["percentage_probability"])


