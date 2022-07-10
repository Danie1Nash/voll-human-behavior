# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from mmpose.apis import inference_top_down_pose_model, init_pose_model


POSE_CONFIG = 'source/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
POSE_CHECKPOINT = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'


def predict(pose_model, img, person_results):
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        img,
        person_results,
        format='xyxy',
        dataset=pose_model.cfg.data.test.type)
    return pose_results


def sort_by_num(path):
    name = path.split('/')[-1]
    num = int(name.split('.')[0])
    return num


@click.command()
@click.argument('video_filepath', type=click.Path(exists=True))
@click.argument('annotation_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('limit_dataset', type=click.INT, required=False)
def main(video_filepath, annotation_filepath, output_filepath, limit_dataset=None):
    if not os.path.exists(f'{output_filepath}/files'):
        os.makedirs(f'{output_filepath}/files')

    # The raw data contains a set of datasets
    # each of which contains the data of a separate video
    datasets = [f.path for f in os.scandir(video_filepath) if f.is_dir()]
    datasets = sorted(datasets, key=sort_by_num)

    # Initialize model for pose detection
    pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT)

    for pdataset in datasets:
        ndataset = int(pdataset.split('/')[-1])
        logging.info(f'Current dataset: {ndataset}')

        data_information = {
            'ndataset': [],
            'nexample': [],
            'poses_path': [],
            'action': [],
            'track_id': []
        }
        # Each dataset containins a set of fragments
        examples = [f.path for f in os.scandir(pdataset) if f.is_dir()]
        for pexample in tqdm(examples[:100]):
            # Set to store the unique IDs of
            # all the people in the frame
            id_frag = set()
            # Bbox + id for each image in fragment
            markup_frag = defaultdict(list)
            # Action for each person in fragment
            action_frag = defaultdict(str)

            # Read exist markup
            nexample = pexample.split('/')[-1]
            with open(f'{annotation_filepath}/{ndataset}/{nexample}/{nexample}.txt') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.split()
                    person_id, xmin, ymin, xmax, ymax, frame, lost, _, _ = map(int, data[:-1])
                    action = str(data[-1])
                    action_frag[person_id] = action
                    id_frag.add(person_id)
                    if lost == 0:
                        markup_frag[frame].append({'bbox': np.array([xmin, ymin, xmax, ymax]), 'track_id': person_id})

            # Predict pose for each images in fragment
            images = sorted(glob(f'{pexample}/*.*'), key=sort_by_num)
            pose_collection = defaultdict(dict)
            for img_path in images:
                img_num = int(img_path.split('/')[-1].split('.')[0])
                for track_id in id_frag:
                    pose_collection[track_id][img_num] = None
                if markup_frag[img_num]:
                    pose_results = predict(pose_model, img_path, markup_frag[img_num])
                    for pose in pose_results:
                        track_id = int(pose['track_id'])
                        pose_collection[track_id][img_num] = pose['keypoints'].tolist()

            # Save result for each person in fragment
            for track_id in pose_collection:
                # print(track_id, str(track_id))
                save_path = (f'{output_filepath}/files/'
                             f'{str(ndataset).zfill(2)}-'
                             f'{str(nexample).zfill(6)}-'
                             f'{str(track_id).zfill(2)}.json')

                with open(save_path, 'w+') as f:
                    json.dump(pose_collection[track_id], f)

                data_information['ndataset'].append(ndataset)
                data_information['nexample'].append(nexample)
                data_information['poses_path'].append(save_path)
                data_information['action'].append(action_frag[track_id])
                data_information['track_id'].append(track_id)

        if limit_dataset is not None:
            if limit_dataset == ndataset:
                break
    pd_data = pd.DataFrame.from_dict(data_information)
    pd_data.to_csv(f'{output_filepath}/data_information.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())

    main()
