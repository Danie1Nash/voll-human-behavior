import click
import logging
from dotenv import find_dotenv, load_dotenv

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def one_hot(data, column):
    one_hot = pd.get_dummies(data[column])
    data = data.drop(column, axis=1)
    data = data.join(one_hot)
    return data


@click.command()
@click.argument('data_inf_path', type=click.Path(exists=True))
@click.argument('dist_map_path', type=click.Path(exists=True))
@click.argument('skeleton_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('output_figure', type=click.Path())
def main(data_inf_path, dist_map_path, skeleton_path, output_filepath, output_figure):
    # Read all dataframe
    data_inf_pd = pd.read_csv(data_inf_path, index_col=0)
    skeleton_pd = pd.read_csv(skeleton_path, index_col=0)
    distant_map_pd = pd.read_csv(dist_map_path, index_col=0)

    # Merge 3 table by index
    all_data_pd = pd.merge(data_inf_pd, skeleton_pd, left_index=True, right_index=True)
    all_data_pd = pd.merge(all_data_pd, distant_map_pd, left_index=True, right_index=True)

    # Delete all rows, which contains None
    join_data = all_data_pd.dropna()

    fig = join_data.action.value_counts().plot(kind='bar', figsize=(12, 12),  fontsize=26).get_figure()
    fig.savefig(f'{output_figure}/all_data_counts.png', bbox_inches='tight')

    # Removing elements from an unbalanced class
    coutnt_border = join_data.action.value_counts().sort_values()[-2]*2
    join_data = join_data.drop(join_data[join_data['action'] == 'standing'].sample(n=coutnt_border).index)

    fig = join_data.action.value_counts().plot(kind='bar', figsize=(12, 12),  fontsize=26).get_figure()
    fig.savefig(f'{output_figure}/crop_data_counts.png', bbox_inches='tight')

    # Removing columns that will not be used further
    join_data = join_data.drop(columns=['ndataset', 'nexample', 'poses_path', 'track_id'])

    # Train-test split
    train_data_pd, val_data_pd = train_test_split(join_data, test_size=0.2, stratify=join_data[['action']])
    train_data_pd = train_data_pd.reset_index(drop=True)
    val_data_pd = val_data_pd.reset_index(drop=True)

    oh_val_data_pd = one_hot(val_data_pd, 'action')
    oh_train_data_pd = one_hot(train_data_pd, 'action')

    oh_val_data_pd.to_csv(f'{output_filepath}/val_data.csv')
    oh_train_data_pd.to_csv(f'{output_filepath}/train_data.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
