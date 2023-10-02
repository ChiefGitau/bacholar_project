import detectron2_helper
from src.helpers import age_sex, data_frame_helper
from tqdm import tqdm

age = age_sex.model_age()
gender = age_sex.model_gender()


def gender_counter(image_list):
    age_list, gender_list = age_sex.get_labels()

    male_counter = 0
    female_counter = 0

    age_spec_list = []


    for el in image_list:
        age_val = age_sex.model_predict(el, age, age_list)
        gender_val = age_sex.model_predict(el, gender, gender_list)

        if gender_val == 'Male':
            male_counter += 1
        else: female_counter += 1;

        string_age = age_val[1:-1].split(',')

        # print('-------------')
        # print(string_age)
        #
        # print(type(age_val))
        #
        # print('-------------')


        aprox = int((int(string_age[0]) + int(string_age[1]))/2)

        age_spec_list.append([gender_val, aprox])


    return male_counter, female_counter, age_spec_list



def human_recon(image_data):
    data = data_frame_helper.get_data_v2()

    # print(data.head)
    # failed_report = []
    for image in tqdm(image_data):
        image_url = image['url']



        prediction, frame, image_value = detectron2_helper.detect_elements(detectron2_helper.get_image_link(image_url))

        image_list = detectron2_helper.get_human_element(image_value, prediction)

        number = len(image_list)

        log_name = image['filePath']
        log_name_split = log_name.split('/')
        generator_name = log_name_split[2]

        if number == 0:
            data.loc[len(data)] = {'image_id': image['url'], 'generator': generator_name, 'number': number,
                                   'male': 0, 'female': 0, "age_details": []}

        else:
            male_count, female_count, age_details = gender_counter(image_list)

            data.loc[len(data)] = {'image_id': image['url'],'generator': generator_name, 'number': number,
                'male': male_count, 'female': female_count, "age_details": age_details}

    return data




