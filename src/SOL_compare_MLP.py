# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Keisuke Shindo
# https://keisukeshindo0.github.io/SOL_feedback_evaluation/
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/
#
# You are free to copy and modify this code for non-commercial purposes, 
# provided that proper credit is given and a link to the license is included.
# Commercial use of this software is strictly prohibited.
#
# Disclaimer: This code is provided "as is", without warranty of any kind,
# express or implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and noninfringement.

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from datetime import datetime
import itertools

# Default values
LogLevel=1
g_IfUnitTestAll=False
g_IfHugeInputMatrix=True
# Version Number
C_Version="Ver_20250323"
g_step=0
C_LearningIterations=1000

import logging
current_time = datetime.now()
formatted_time = current_time.strftime("%Y%m%d_%H%M")


import os
# ユーザーディレクトリを取得
user_dir = os.path.expanduser("~")

# 出力ディレクトリを作成
log_dir = os.path.join(user_dir, "log_files")
os.makedirs(log_dir, exist_ok=True)

# ログファイルのパス
log_file_path = os.path.join(log_dir, 'SOL_NeuralNetworkCompetitor'+formatted_time+'.log')

logging.basicConfig(
    filename=log_file_path,          # 出力するファイル名
    filemode='w',                 # 'a'は追記モード、'w'は上書き
    level=logging.INFO,           # INFOレベル以上のログを記録
    format='%(message)s'
)
#    format='%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format='%(message)s')

def Log(*args, sep=' ', end='\n'):
    message = "step"+str(g_step)+" : "+sep.join(map(str, args)) + end.strip()
    logging.info(message)
    print(message)




##########################################################################
# Data generator
##########################################################################

class SOLNetwork:
    
    @staticmethod
    def GenerateMatrix(dims, steps, random_ratio=0.1, priority_ratio=0.6):
        """
        冒頭4要素を優先的にランダム変動させ、それ以外の要素の変動を最小限にするベクトル列を生成。

        Args:
            dims (int): ベクトルの次元数。
            steps (int): 生成するステップ数。
            random_ratio (float): 完全ランダムにする確率。
            priority_ratio (float): 先頭4要素を変更する確率。

        Returns:
            list: 生成されたベクトルのリスト。
        """
        current_vector = torch.zeros(dims, dtype=torch.int32)
        sequence = [current_vector.tolist()]

        priority_indices = list(range(min(4, dims)))  # 先頭4要素のインデックス
        other_indices = list(range(4, dims))  # それ以外のインデックス

        for _ in range(steps - 1):
#            if random.random() < random_ratio:
#                new_vector = torch.randint(0, 2, (dims,), dtype=torch.int32)  # 完全ランダム
#            else:
            if True:
                new_vector = current_vector.clone()
                if random.random() < priority_ratio or not other_indices:
                    # 先頭4要素のどれかを変更
                    idx = random.choice(priority_indices)
                else:
                    # それ以外の要素を低頻度で変更
                    idx = random.choice(other_indices)

                new_vector[idx] = 1 - new_vector[idx]  # 反転
            
            sequence.append(new_vector.tolist())
            current_vector = new_vector

        return sequence

# Global variables

class DataGenerator:    
    @staticmethod
    def GenerateData():    
        # Generate all combinations of data
        X=None
        Y=None
        XSize=0
        YSize=0

        if g_IfHugeInputMatrix:
            XSize=32
            X=SOLNetwork.GenerateMatrix(XSize,C_LearningIterations)
            X = torch.tensor(X, dtype=torch.float32)
            X_int = X.int()
        else:
            XSize=4
            X = list(itertools.product([0, 1], repeat=XSize))

            X = torch.tensor(X, dtype=torch.float32)
            X_int = X.int()


        Log('Inputs: ')
        Log(X_int)


        # Logic operations
        YEquation=[]
        YEquation.append(X_int[:, 0] & X_int[:, 1])
        YEquation.append(X_int[:, 2] | X_int[:, 3])
        YEquation.append(X_int[:, 1] ^ X_int[:, 3])
        YEquation.append((~X_int[:, 0]&1) & X_int[:, 1])
        YEquation.append((~(X_int[:, 0]& X_int[:, 1])&1)) 
        YEquation.append((~X_int[:, 2]&1) | X_int[:, 3])
        YEquation.append((~X_int[:, 0]&1) ^ X_int[:, 1])
        YEquation.append((~(X_int[:, 2]& X_int[:, 3])&1)) 
        YEquation.append((X_int[:, 0] & X_int[:, 1] & X_int[:, 2] & X_int[:, 3]))
        YEquation.append((X_int[:, 0] | X_int[:, 1] | X_int[:, 2] | X_int[:, 3]))
        YEquation.append((X_int[:, 0] ^ X_int[:, 1] ^ (~X_int[:, 2]&1) ^ X_int[:, 3]))
        YEquation.append((X_int[:, 0] | X_int[:, 1]) & (X_int[:, 2] | X_int[:, 3]))
        YEquation.append((X_int[:, 0] | (~X_int[:, 1])&1) | (X_int[:, 2] | X_int[:, 3]))
        YEquation.append((X_int[:, 0] ^ X_int[:, 1]) | (X_int[:, 2] ^ X_int[:, 3]))
        YEquation.append((X_int[:, 0] ^ (~X_int[:, 1]&1)) & (X_int[:, 2] ^ (~X_int[:, 3]&1)))
        YEquation.append((X_int[:, 0] | X_int[:, 1]) ^ ((~X_int[:, 2]&1) & X_int[:, 3]))
        ReferenceEq=[
        "X_int[:, 0] & X_int[:, 1]",
        "X_int[:, 2] | X_int[:, 3]",
        "X_int[:, 1] ^ X_int[:, 3]",
        "(~X_int[:, 0]&1) & X_int[:, 1]",
        "(~(X_int[:, 0]& X_int[:, 1])&1)",
        "(~X_int[:, 2]&1) | X_int[:, 3]",
        "(~X_int[:, 0]&1) ^ X_int[:, 1]",
        "~(X_int[:, 2]& X_int[:, 3])&1) ",
        "X_int[:, 0] & X_int[:, 1] & X_int[:, 2] & X_int[:, 3]",
        "X_int[:, 0] | X_int[:, 1] | X_int[:, 2] | X_int[:, 3]",
        "X_int[:, 0] ^ X_int[:, 1] ^ (~X_int[:, 2]&1) ^ X_int[:, 3]",
        "(X_int[:, 0] | X_int[:, 1]) & (X_int[:, 2] | X_int[:, 3])",
        "X_int[:, 0] | (~X_int[:, 1])&1) | (X_int[:, 2] | X_int[:, 3]",
        "(X_int[:, 0] ^ X_int[:, 1]) | (X_int[:, 2] ^ X_int[:, 3])",
        "(X_int[:, 0] ^ (~X_int[:, 1]&1)) & (X_int[:, 2] ^ (~X_int[:, 3]&1))",
        "(X_int[:, 0] | X_int[:, 1]) ^ ((~X_int[:, 2]&1) & X_int[:, 3])",
        ]
        YVector=[]
        for eachY in YEquation:
            YVector.append(eachY.float())

        Y = torch.stack(YVector, dim=1)
        YSize=Y.size(1)
        #YSize=12 # Temporary
        #X = X.float()
        X = 2 * X - 1  # 0,1 を -1,1 に変換
        Y = Y.float()
        return X,Y,XSize,YSize

class LogicMLP(nn.Module):
    def __init__(self, XSize, YSize, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(XSize, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, YSize)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # 出力を 0-1 に制限
    
from torch.utils.data import DataLoader, TensorDataset

# Example usage
if __name__ == "__main__":

    X,Y,XSize,YSize=DataGenerator.GenerateData()

    #lr=0.01
    lr=0.001
    batch_size=16
    #batch_size=32
    #_hidden_dim=128
    _hidden_dim=64
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()  # ロジット付きバイナリ交差エントロピー損失
    # Hyperparameters
    model = LogicMLP(XSize=32, YSize=16,hidden_dim=_hidden_dim)  # YSize は論理式の数

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-5)

    start_time = time.time()

    num_epochs = 3500
    for epoch in range(num_epochs):
        g_step=epoch
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_Y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = model(X)
            #predictions = torch.sigmoid(logits)
            predictions=logits
            preds = (predictions > 0.5).float()
            accuracy = (preds == Y).float().mean().item()

        Log('Epoch:', epoch)
        Log('---------------------------------')    
        Log('Reference:')
        Log(Y)
        Log('Predictions:')
        Log(predictions)
        Log('Accuracy:', accuracy)
        Log('Loss:', loss.item())

    Log("--------------------------------------------")
    Log("Learning ended")

    bitwise_accuracy = (preds == Y).float().mean(dim=0)  # 各列の平均
    Log("Bitwise accuracy", bitwise_accuracy)

    end_time = time.time()
    elapsed_time = end_time - start_time
    Log("Learning epochs ",num_epochs)
    Log(f"Elapsed time of learning: {elapsed_time}sec")

    Log("Current time",datetime.now()," Version",C_Version)
    Log("Log file created to",log_file_path)
