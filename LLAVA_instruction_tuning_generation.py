import time
from huggingface_hub.inference._text_generation import FinishReason

from pycocotools.coco import COCO

import parse_responses
from text_generation import Models, APITypes, ResponseTypes, TextGenerator
import json

instances_annotation_file = 'annotations/instances_train2017.json'
coco_instances = COCO(instances_annotation_file)

captions_annotations_file = 'annotations/captions_train2017.json'
coco_caps = COCO(captions_annotations_file)

# get all the image ids from the file
image_ids = coco_instances.getImgIds()
num_images = len(image_ids)

# map of category_ids to category name
cats = coco_instances.loadCats(coco_instances.getCatIds())
cat_id_2_name = {cat['id']: cat['name'] for cat in cats}

text_generator = TextGenerator(model=Models.MISTRAL, api_type=APITypes.INFERENCE_API, max_new_tokens=750)

# response_types = [ResponseTypes.COMPLEX_REASONING, ResponseTypes.CONVERSATION, ResponseTypes.DETAIL_DESCRIPTION ]
response_types = [ResponseTypes.CONVERSATION]

image_ids = [33471, 52846, 334872, 319154, 398214]
for image_id in image_ids:
    img_info = coco_instances.loadImgs(image_id)[0]
    img_width = img_info['width']
    img_height = img_info['height']
    print(image_id)

    instance_ann_ids = coco_instances.getAnnIds(imgIds=image_id)
    instance_anns = coco_instances.loadAnns(instance_ann_ids)

    ## normalize the bounding boxes
    normalized_bboxs = []
    for ann in instance_anns:
        bbox_x, bbox_y, bbox_width, bbox_height = ann['bbox']
        normalized_bbox = [round(bbox_x / img_width, 3), round(bbox_y / img_height, 3),
                           round((bbox_x + bbox_width) / img_width, 3), round((bbox_y + bbox_height) / img_height, 3)]

        category = cat_id_2_name[ann['category_id']]
        normalized_bboxs.append(f'{category}: {normalized_bbox}')

    # get the image captions
    caps_ann_ids = coco_caps.getAnnIds(imgIds=image_id)
    caps_anns = coco_caps.loadAnns(caps_ann_ids)

    bboxs = '\n'.join(normalized_bboxs)
    captions = '\n'.join(ann['caption'] for ann in caps_anns)

    parsed_responses = {'image_id':image_id}
    for response_type in response_types:
        # TODO: pass both the captions and the bboxs to the text generator
        if response_type == ResponseTypes.CONVERSATION:
            query = captions
        else:
            query = captions + '\n\n' + bboxs

        generated_text, finish_reason = text_generator.generate(query, response_type)
        # generated_text, finish_reason = text_generator.generate_complex_reasoning_pruned(captions, bboxs)

        with open(f"datasets/Mistral/raw/{response_type.value}.json", "a") as jf:
            json.dump({image_id: generated_text}, jf, indent=2)
            jf.write(',\n')

        if finish_reason == FinishReason.Length:
            with open(f"datasets/Mistral/error_log.json", "a") as jf:
                json.dump({'image_id':image_id,'text':generated_text,'error':'finish length'}, jf, indent=2)
                jf.write(',\n')
            continue

        parsed_response = generated_text if (response_type == ResponseTypes.DETAIL_DESCRIPTION)\
            else parse_responses.parse_conversation(generated_text)

        if parsed_response:
            parsed_responses[response_type.value]=parsed_response
        # otherwise something went wrong parsing the text
        else:
            with open(f"datasets/Mistral/error_log.json", "a") as jf:
                json.dump({'image_id':image_id,'text':generated_text,'error':'parsing'}, jf, indent=2)
                jf.write(',\n')

    print(parsed_responses)
    if len(parsed_responses) > 1:
        with open("datasets/Mistral/json/all_responses.json", "a") as jf:
            json.dump(parsed_responses, jf, indent=2)
            jf.write(',\n')