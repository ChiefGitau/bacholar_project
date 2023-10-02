import detectron2,cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
from skimage import io


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"  # we use a CPU Detectron copy
# create predictor
predictor = DefaultPredictor(cfg)


def detect_elements(img):

  image = cv2.cvtColor(img, cv2.IMREAD_COLOR)
  # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # predict categories
  return predictor(image), img, image

def vis_image(image):
    plt.imshow(image)
    plt.show()


def vis_prediction(image, output):

  v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2)
  out = v.draw_instance_predictions(output["instances"].to("cpu"))

  plt.figure()
  plt.imshow(out.get_image()[..., ::-1][..., ::-1])
  plt.show()


def get_human_element(image, output):
    matches = zip(output['instances'].get('pred_classes'), output['instances'].get('pred_boxes'))

    image_list = []

    for l, j in matches:
        # skip all categories which do not correspond to people
        if int(l) != 0: continue
        # get bounding box for person
        i = [int(k) for k in j]

        # # crop the original image using the bounding box
        img = image[i[1]:i[3], i[0]:i[2]]
        image_list.append(image[i[1]:i[3], i[0]:i[2]])  # crop to bb
        # plt.imshow(img)
        # plt.show()

    return image_list


def vis_human_element(image, output):
    matches = zip(output['instances'].get('pred_classes'), output['instances'].get('pred_boxes'))

    for l, j in matches:
        # skip all categories which do not correspond to people
        if int(l) != 0: continue
        # get bounding box for person
        i = [int(k) for k in j]

        # # crop the original image using the bounding box
        img = image[i[1]:i[3], i[0]:i[2]]  # crop to bb
        plt.imshow(img)
        plt.show()

def get_image_link(img_url):

    return io.imread(img_url)

def get_img_local(img_loc):
    return cv2.imread(img_loc)