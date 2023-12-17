import argparse
import pickle
import sys

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from data.data_loader import LoadDataAndLabel
from tqdm import tqdm

from models.model import MLP
from sklearn.ensemble import RandomForestRegressor


def train_MLP(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_people = [100] * 24
    time = [i for i in range(24)]
    people = [5, 4, 2, 1, 0, 10, 100, 103, 356, 433, 201, 78, 874, 532, 233, 171, 92, 888, 917, 537, 207, 312, 103, 53]
    data_dict = {"base_people": base_people, "time": time, "people": people}
    data = pd.DataFrame(data_dict)

    data_set = LoadDataAndLabel(data)

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                              collate_fn=LoadDataAndLabel.collate_fn)

    model = MLP(inp=args.inp, oup=1)
    model = model.to(device)

    lossfunction = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    lf = lambda x: (1 - x / args.epochs) * (1.0 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_loader_bar = tqdm(data_loader, file=sys.stdout)
    model.train()
    for epoch in range(args.epochs):
        train_loader_bar.reset()
        for batch in train_loader_bar:
            data, target = batch
            optimizer.zero_grad()
            outputs = model(data.to(device))
            outputs = outputs.squeeze(dim=1)
            loss = lossfunction(outputs, target.float().to(device))

            loss.backward()
            optimizer.step()
            scheduler.step()

            # train_loader_bar.desc = "epoch {}, loss: {}".format(epoch, loss.item())
            train_loader_bar.set_postfix(epoch=epoch, loss=loss.item())
        print(loss.item())
    torch.save(model.state_dict(), f="MLP.pth")


def train_randomforest():
    rfr = RandomForestRegressor(200)
    base_people = [100] * 24
    time = [i for i in range(24)]
    people = [5, 4, 2, 1, 0, 10, 100, 103, 356, 433, 201, 78, 874, 532, 233, 171, 92, 888, 917, 537, 207, 312, 103, 53]
    time = np.reshape(time, [-1, 1])
    # 创建一个管道（Pipeline）实例，里面包含标准化方法和随机森林模型估计器
    pipeline = make_pipeline(StandardScaler(), rfr)
    # 设置交叉验证折数cv 表示使用带有十折的StratifiedKFold，再把管道和数据集传到交叉验证对象中

    scores = cross_val_score(pipeline, X=time, y=people, n_jobs=1)

    pipeline.fit(time, people)
    print('Cross Validation accuracy scores: %s' % scores)
    print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    save_path = "random_forest_reg_model.pkl"

    with open(save_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print("finish training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--inp", default=2,
                        help='data path')
    parser.add_argument("--batch_size", default=12)
    parser.add_argument("--epochs", default=300)

    parser.add_argument("--lr", default=0.01)
    parser.add_argument("--lrf", default=0.009)

    args = parser.parse_args()

    train_MLP(args)
