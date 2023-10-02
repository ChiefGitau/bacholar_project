import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

caffe_root = os.path.abspath('/home/hidde/Desktop/git/Vu/Seeing_the_words/paper_model')
import sys
# sys.path.insert(0, caffe_root + 'python')
import caffe

def model_gender():
    gender_net_pretrained = caffe_root + '/gender_net.caffemodel'
    gender_net_model_file = caffe_root + '/deploy_gender.prototxt'
    return caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                                  mean=setup_mean(),
                                  channel_swap=(2, 1, 0),
                                  raw_scale=255,
                                  image_dims=(256, 256))
def model_age():
    age_net_pretrained = caffe_root + '/age_net.caffemodel'
    age_net_model_file = caffe_root + '/deploy_age.prototxt'
    return caffe.Classifier(age_net_model_file, age_net_pretrained,
                               mean=setup_mean(),
                               channel_swap=(2, 1, 0),
                               raw_scale=255,
                               image_dims=(256, 256))

def setup_mean():
    mean_filename = caffe_root + '/age_gender_mean.binaryproto'
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    return caffe.io.blobproto_to_array(a)[0]


def get_labels():
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Male', 'Female']

    return age_list, gender_list


def convert_image(image):
    #image1 = cv2.imread(caffe_root + 'examples/images/cat.jpg')
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = image1 / 255.
    return image1


def model_predict(image, model, age_list):
    # %%
    prediction = model.predict([convert_image(image)])

    return age_list[prediction[0].argmax()]


