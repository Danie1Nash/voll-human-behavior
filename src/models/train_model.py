import os
import logging
from tqdm import tqdm
from typing import List
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ignite.metrics import Accuracy, Precision, Recall, Fbeta, Loss

import click
from dotenv import load_dotenv

import dvc.api
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


params = dvc.api.params_show()['train_model']

load_dotenv()
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(remote_server_uri)


class CustomDataset(Dataset):
    def __init__(self, skeleton_img_path, dist_map_path, labels_y):
        self.skeleton_img_path = skeleton_img_path
        self.dist_map_path = dist_map_path
        self.labels_y = labels_y

    def __getitem__(self, index):
        skeleton_img = cv2.imread(self.skeleton_img_path[index])
        dist_map = cv2.imread(self.dist_map_path[index])

        skeleton_img = normalize_image(skeleton_img)
        skeleton_img_tensor = torch.tensor(skeleton_img, dtype=torch.float)

        dist_map = normalize_image(dist_map)
        dist_map_tensor = torch.tensor(dist_map, dtype=torch.float)

        y = torch.tensor(self.labels_y[index], dtype=torch.long)
        return [skeleton_img_tensor, dist_map_tensor], y

    def __len__(self):
        return self.skeleton_img_path.shape[0]


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(12, 36, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(49284+48, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, num_classes)

    def forward(self, input1, input2):
        out1 = self.layer1(input1)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)
        out1 = out1.reshape(out1.size(0), -1)
        out1 = self.drop_out(out1)

        out2 = self.layer1(input2)
        out2 = self.layer2(out2)
        out2 = out2.reshape(out2.size(0), -1)
        out2 = self.drop_out(out2)

        out = torch.cat((out1.view(out1.size(0), -1), out2.view(out2.size(0), -1)), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class BestModelCallback:
    # metric - metric to monitor
    # mode - min or max
    def __init__(self, model, metric, mode):
        self.metric = metric

        if mode not in ["min", "max"]:
            raise Exception("mode should be one max or min")

        self.mode = mode
        if self.mode == "min":
            self.best = float('Inf')
        else:
            self.best = -float('Inf')

        self.model = model
        # self.path = path
        self.log = None

    def save_best(self, logs, model):
        if self.mode == "min":
            if logs[self.metric] < self.best:
                self.best = logs[self.metric]
                self.model = model
                self.logs = logs
                # torch.save(self.model.state_dict(), self.path)
                # print(f"New Model has been saved: {self.metric} = {self.best}")
        else:
            if logs[self.metric] > self.best:
                self.best = logs[self.metric]
                self.model = model
                self.logs = logs
                # torch.save(self.model.state_dict(), self.path)
                # print(f"New Model has been saved: {self.metric} = {self.best}")


class MetricsGroup:
    def __init__(self, metrics_dict):
        self.metrics = metrics_dict

    def update(self, output):
        for name, metric in self.metrics.items():
            metric.update(output)

    def compute(self):
        output = {}
        for name, metric in self.metrics.items():
            output[name] = metric.compute()
        return output

    def reset(self):
        for name, metric in self.metrics.items():
            metric.reset()


def print_logs(epoch, logs, start_time, end_time):
    print(f'epoch: {epoch}')
    print(f'start time: {start_time}')
    print(f'end time: {end_time}')
    for (key, val) in logs.items():
        print(f'{key}: {val}')
    print('-------------------------------------------\n')


def init_metrics(criterion):
    p = Precision(average=False)
    r = Recall(average=False)
    m_group = MetricsGroup({
        'loss': Loss(criterion),
        "accuracy": Accuracy(),
        "precision": p,
        "recall": r,
        "f1": Fbeta(beta=1.0, average=False, precision=p, recall=r)
    })
    return m_group


def update_metrics(pred, target, m_group):
    softm = nn.Softmax(dim=1)
    soft_pred = softm(pred)
    m_group.update((soft_pred, torch.argmax(target, dim=1)))


def update_logs(mode, m_group, logs):
    scores = m_group.compute()
    prefix = 'val_' if mode == 'val' else ''
    logs[prefix + 'accuracy'] = scores['accuracy']
    logs[prefix + 'f1'] = scores['f1'].mean().item()
    logs[prefix + 'loss'] = scores['loss']


def update_history(logs, history):
    for key, val in logs.items():
        history[key].append(val)


def normalize_image(image):
    image = image.astype('float32')
    image /= 255
    image = image.transpose(2, 0, 1)
    return image


def create_dataset(csv_path):
    df = pd.read_csv(csv_path, index_col=[0])
    dist_map_path = np.array([str(path) for path in df.pop('distant_map_path')])
    skeleton_img_path = np.array([str(path) for path in df.pop('skeleton_path')])
    label2class = df.columns
    labels = df.to_numpy().astype('float32')
    num_classes = labels.shape[1]
    return CustomDataset(skeleton_img_path, dist_map_path, labels), num_classes, label2class


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path(), nargs=2)
def train(input_paths: List[str], output_path: List[str]):
    """
    Train the model and log params, metrics and artifacts in MLflow
    :param input_paths: train (for [0]) and test (for [1]) dataframes
    :param output_path: model (for [0]) and score (for [1]) artifact's path
    :return:
    """
    logging.info('Сurrent stage: start')
    mlflow.set_experiment("lenet")
    with mlflow.start_run():
        # Input params
        logging.info('Сurrent stage: create dataloaders')
        train_dataset, num_classes, label2class = create_dataset(input_paths[0])
        val_dataset, num_classes, label2class = create_dataset(input_paths[1])
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=True)

        logging.info('Сurrent stage: create model')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = ConvNet(num_classes=num_classes)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])

        best_model = BestModelCallback(model=model, metric='val_loss', mode='min')
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.7, verbose=True)

        m_group = init_metrics(lambda x, y: F.nll_loss(torch.log(x), y))
        logs = dict()
        history = defaultdict(list)

        logging.info('Start learning models')
        for epoch in range(0, params["epochs"]):
            logging.info(f'Сurrent stage: epoch {epoch+1}/{params["epochs"]}')

            start_time = datetime.now()
            model.train()

            # Training
            for x_batch, y_batch in tqdm(train_loader):
                skeleton_batch, dist_batch, y_batch = x_batch[0].to(device), x_batch[1].to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(skeleton_batch, dist_batch)
                loss = criterion(y_pred, torch.argmax(y_batch, dim=1))
                loss.backward()
                optimizer.step()
                update_metrics(y_pred, y_batch, m_group)
            update_logs('train', m_group, logs)
            m_group.reset()

            # Validation
            val_loss = 0
            with torch.no_grad():
                model.eval()
                for x_val, y_val in tqdm(val_loader):
                    skeleton_val, dist_val, y_val = x_val[0].to(device), x_val[1].to(device), y_val.to(device)                                   
                    val_pred = model(skeleton_val, dist_val)
                    val_loss += criterion(val_pred, torch.argmax(y_val, dim=1)).item()
                    update_metrics(val_pred, y_val, m_group)
            update_logs('val', m_group, logs)
            m_group.reset()

            update_history(logs, history)

            lr_scheduler.step(val_loss / len(val_loader))
            best_model.save_best(logs, model)
            end_time = datetime.now()
            print_logs(logs=logs, epoch=epoch, start_time=start_time, end_time=end_time)

            mlflow.log_metrics(logs)

        torch.save(best_model.model, f"{output_path[0]}/model.pth")

        # Create confusion matrix
        nb_classes = num_classes
        confusion_matrix = np.zeros((nb_classes, nb_classes))
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(val_loader):
                inputs1 = inputs[0].to(device)
                inputs2 = inputs[1].to(device)
                classes = classes.to(device)
                outputs = model(inputs1, inputs2)
                _, preds = torch.max(outputs, 1)
                _, classes = torch.max(classes, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        plt.figure(figsize=(15, 10))
        class_names = list(label2class.values)
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f"{output_path[0]}/confusion_matrix.jpg")

        # Create history accuracy
        plt.figure(figsize=(10, 7))
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        plt.savefig(f"{output_path[0]}/accuracy_graph.jpg")

        # Create history loss
        plt.figure(figsize=(10, 7))
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        plt.savefig(f"{output_path[0]}/loss_graph.jpg")

        mlflow.log_params(params)

        # mlflow.log_metrics(best_model.logs)
        mlflow.log_artifact(f"{output_path[0]}/confusion_matrix.jpg")
        mlflow.log_artifact(f"{output_path[0]}/accuracy_graph.jpg")
        mlflow.log_artifact(f"{output_path[0]}/loss_graph.jpg")

        input_schema = Schema([
            TensorSpec(np.dtype(np.uint8), (-1, 28, 28, 1)),
        ])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        mlflow.pytorch.log_model(best_model.model, "model", signature=signature)


if __name__ == "__main__":
    train()
