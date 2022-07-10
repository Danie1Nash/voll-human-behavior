import click
import logging
from dotenv import find_dotenv, load_dotenv

import os
import json
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd

DISTANCE_MAP_SIZE = (60, 18)
AXIS_NUMBER = 2
WINDOW_SIZE = 3
IMAGE_BORDER = 10
POINT_NUMBER = 17
CENTER_BONE = [11, 12]


def norm_vals(x):
    x = x-np.min(x)
    x = x/np.max(x)
    return x


def get_distant_map(data, left, right):
    image = None
    for i in range(left, right+1):
        if data[i] is None:
            continue
        points = data[i][:, :AXIS_NUMBER]
        center_point = (points[CENTER_BONE[0], :] + points[CENTER_BONE[1], :])/2
        points -= center_point
        points = np.append(points, np.zeros((points.shape[0], 1)), axis=1)
        if image is None:
            image = np.atleast_3d(points)
        else:
            image = np.append(image, np.atleast_3d(points), axis=2)
    if image is None:
        return None
    image = np.swapaxes(image, 1, 2)
    for i in range(POINT_NUMBER):
        for j in range(AXIS_NUMBER):
            image[i, :, j] = norm_vals(image[i, :, j])
    resize_image = cv2.resize(image*255, (POINT_NUMBER, WINDOW_SIZE*2+1), interpolation=cv2.INTER_AREA)
    return(resize_image)


@click.command()
@click.argument('data_information_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(data_information_path, output_filepath):
    if not os.path.exists(f"{output_filepath}/distant_map"):
        os.makedirs(f"{output_filepath}/distant_map")

    result = {'index': [], 'distant_map_path': []}

    data = pd.read_csv(data_information_path)
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        nexample = row["nexample"]
        ndataset = row["ndataset"]
        track_id = row["track_id"]

        with open(row["poses_path"]) as f:
            poses = json.load(f)

        for k, v in poses.items():
            if v is not None:
                poses[k] = np.array(v)

        poses = {int(k): v for k, v in poses.items()}

        left = nexample - WINDOW_SIZE
        right = nexample + WINDOW_SIZE

        distant_map = get_distant_map(poses, left, right)

        if distant_map is not None:
            save_path = (f'{output_filepath}/distant_map/'
                         f'{str(ndataset).zfill(2)}-'
                         f'{str(nexample).zfill(6)}-'
                         f'{str(track_id).zfill(2)}.png')
            cv2.imwrite(save_path, distant_map)
        else:
            save_path = None

        result['index'].append(index)
        result['distant_map_path'].append(save_path)

    pd_result = pd.DataFrame.from_dict(result)
    pd_result = pd_result.set_index('index')
    pd_result.to_csv(f'{output_filepath}/distant_map.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
