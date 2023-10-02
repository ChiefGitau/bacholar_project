from imagekitio import ImageKit
import os

from imagekitio.models.ListAndSearchFileRequestOptions import ListAndSearchFileRequestOptions

location_list = ['stabilityai', 'SG161222', 'runwayml', 'prompthero', 'nitrosocke', 'mid_journey', 'dall_e_2', 'dream', 'CompVis', 'base']

def setup():
    return ImageKit(
        private_key='',
        public_key='',
        url_endpoint='https://ik.imagekit.io/seeingthewords'
    )


def get_image(path):
    return setup().url({
        "path": path,
        "url_endpoint": "https://ik.imagekit.io/seeingthewords/"
    })

def get_file_details(path):
    return setup().list_files(options=ListAndSearchFileRequestOptions(path=path,
                                                                      search_query = 'format ="jpg"'
                                                                      )).response_metadata.raw
def get_target_data(target):
    temp = []
    for loc in location_list:
       data = setup().list_files(options=ListAndSearchFileRequestOptions(path = 'data/'+loc+ '/' + target)).response_metadata.raw
       print(f'{loc=} contains  {len(data)=}')
       for image in data:
               temp.append(image)
    return temp

