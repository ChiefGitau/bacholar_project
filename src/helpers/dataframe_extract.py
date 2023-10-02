import pandas as pd
from tqdm import tqdm
import caffe
import cv2
import numpy as np
from skimage import io

mean_file = 'ilsvrc_2012_mean.npy'
deploy_path = 'sentiment_deploy.prototxt'
caffemodel_path = 'twitter_finetuned_test4_iter_180.caffemodel'
net_pred = caffe.Classifier(deploy_path,
                            caffemodel_path,
                            mean=np.load(mean_file).mean(1).mean(1),
                            image_dims=(256, 256),
                            channel_swap=(2, 1, 0),
                            raw_scale=255)




model_loc = 'twitter_finetuned_test4_iter_180_conv.caffemodel'
model_deploy ='sentiment_maps/sentiment_fully_conv_deploy.prototxt'


model =caffe.Net(model_deploy,
                           model_loc,  caffe.TEST)

transformer = caffe.io.Transformer({'data': model.blobs['data'].data.shape})
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

def get_image_link(img_url):

    return io.imread(img_url)

def get_map_dif(np_base, image_url_image):
    out_2 = model.forward_all(data=np.asarray([transformer.preprocess('data', image_url_image)]))

    # print(out)
    el = out_2['prob'][0][1]

    total_val = 0

    for i, ls in enumerate(el):
        test_av = 0
        for j, el_sub in enumerate(ls):
            test_av += abs(el_sub - np_base[i][j])

        total_val += (test_av / len(ls))

    return (total_val / len(el))

def get_data():
    columns_section = ["image_id", "generator", "predict_score",
                       "map_diff_score_base_0", "map_diff_score_base_1",
                       "map_diff_score_base_2", "map_diff_score_base_3","map_diff_score_base_4"]
    return pd.DataFrame(columns=columns_section)

def get_base(image_data):
    base_list = []
    for image in image_data:
        if 'base' in image['url']:
            base_list.append(image['url'])
    return base_list

def sent_recon(image_data):
    data =get_data()
    base = get_base(image_data)
    np_base = []

    for base_image in base:
        out = model.forward_all(data=np.asarray([transformer.preprocess('data',get_image_link(base_image) )]))
        np_base.append(out['prob'][0][1])


    for image in tqdm(image_data):
        image_url = image['url']

        image_url_image = get_image_link(image_url)


        prediction = net_pred.predict([image_url_image])

        predict_prompt = []

        for base_image in np_base:
            predict_prompt.append(get_map_dif(base_image, image_url_image))


        log_name = image['filePath']
        log_name_split = log_name.split('/')
        generator_name = log_name_split[2]

        data.loc[len(data)] = {'image_id': image['url'],'generator': generator_name, 'predict_score': prediction,
                               "map_diff_score_base_0" : predict_prompt[0], "map_diff_score_base_1": predict_prompt[1],
        "map_diff_score_base_2":predict_prompt[2], "map_diff_score_base_3":predict_prompt[3], "map_diff_score_base_4":predict_prompt[4]
        }

    return data