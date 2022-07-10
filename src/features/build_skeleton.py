import click
import logging
from dotenv import find_dotenv, load_dotenv

import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


DISTANCE_MAP_SIZE = (60, 18)
AXIS_NUMBER = 2
WINDOW_SIZE = 3
IMAGE_BORDER = 10
POINT_NUMBER = 17
BONE_LIST = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
             [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
             [8, 10]]
COLOR_MAP = plt.get_cmap('YlGn')


def plot_im_skeleton(skeleton, ax, color):
    x = skeleton[:, 0]
    y = skeleton[:, 1]
    for bone in BONE_LIST:
        ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], color = color[:3])


def get_skeleton(data, left, right):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    xmin, xmax = None, None
    ymin, ymax = None, None

    colors = COLOR_MAP(np.linspace(0.0, 1, right-left+1))

    for i in range(left, right+1):
        if data[i] is not None:
            if xmin is None:
                xmin, xmax = data[i][:, 0].min(), data[i][:, 0].max()
                ymin, ymax = data[i][:, 1].min(), data[i][:, 1].max()
            else:
                xmin, xmax = min(xmin, data[i][:, 0].min()), max(xmax, data[i][:, 0].max())
                ymin, ymax = min(ymin, data[i][:, 1].min()), max(ymax, data[i][:, 1].max())
            plot_im_skeleton(data[i], ax, colors[i-left])

    if xmin is None:
        plt.close()
        return None

    if xmax-xmin > ymax-ymin:
        delta = ((xmax-xmin) - (ymax-ymin))/2
        ymax += delta
        ymin -= delta
    else:
        delta = ((ymax-ymin) - (xmax-xmin))/2
        xmax += delta
        xmin -= delta

    xmin -= IMAGE_BORDER
    xmax += IMAGE_BORDER
    ymin -= IMAGE_BORDER
    ymax += IMAGE_BORDER

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.gca().invert_yaxis()
    ax.set_axis_off()
    plt.close()
    return fig


@click.command()
@click.argument('data_information_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(data_information_path, output_filepath):
    if not os.path.exists(f"{output_filepath}/skeleton"):
        os.makedirs(f"{output_filepath}/skeleton")

    result = {'index': [], 'skeleton_path': []}

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

        skeleton_img = get_skeleton(poses, left, right)

        if skeleton_img is not None:
            save_path = (f'{output_filepath}/skeleton/'
                         f'{str(ndataset).zfill(2)}-'
                         f'{str(nexample).zfill(6)}-'
                         f'{str(track_id).zfill(2)}.png')
            skeleton_img.savefig(save_path)
        else:
            save_path = None

        result['index'].append(index)
        result['skeleton_path'].append(save_path)

    pd_result = pd.DataFrame.from_dict(result)
    pd_result = pd_result.set_index('index')
    pd_result.to_csv(f'{output_filepath}/skeleton.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
