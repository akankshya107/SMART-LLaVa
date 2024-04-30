#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from PIL import Image
import os
import csv
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# In[2]:


device = 0
print(device)


# In[3]:


DATASET_DIR = "/home/ritaban/smart/SMART101-release-v1/SMART101-Data/"


# In[4]:


def read_csv(csvfilename, puzzle_id):
    import csv
    qa_info = []
    with open(csvfilename, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            row["puzzle_id"] = str(puzzle_id)
            if len(row["A"]) == 0:
                row["A"] = "A"
                row["B"] = "B"
                row["C"] = "C"
                row["D"] = "D"
                row["E"] = "E"
            qa_info.append(row)
    return qa_info

SEQ_PUZZLES = [16, 18, 35, 39, 63, 100]
SIGNS = np.array(["+", "-", "x", "/"])
MAX_DECODE_STEPS = 10

def get_puzzle_class_info(puzzle_ids, icon_class_ids):
    #    global SEQ_PUZZLES, puzzle_diff_str, puzzle_diff
    puzzle_classes = {}
    for puzzle_id in puzzle_ids:
        puzzle_root = puzzle_id
        csv_file = "puzzle_%s.csv" % (puzzle_id)
        qa_info = read_csv(os.path.join(DATASET_DIR, puzzle_root, csv_file), puzzle_id)

        pid = int(puzzle_id)
        if pid not in SEQ_PUZZLES:
            num_classes = np.array([get_val(qa, qa["Answer"], {}, icon_class_ids) for qa in qa_info]).max() + 1
        else:
            if pid in [16, 39, 100]:
                num_classes = 26 + 1  # if the output is a string of numbers, and the max classes is - max val.
            elif pid in [18, 35]:
                num_classes = 5 + 1  # the minus one is for end of items.
            elif pid in [63]:
                num_classes = np.array([get_val(qa, qa["Answer"], {}, icon_class_ids).max() for qa in qa_info]).max() + 1
        puzzle_classes[str(puzzle_id)] = num_classes
    return puzzle_classes

def get_icon_dataset_classes(icon_path):
    """returns the classes in ICONs-50 dataset"""
    with open(icon_path, "r") as f:
        icon_classes = f.readlines()
    return [ii.rstrip() for ii in icon_classes]

def str_replace(ans):
    ans = ans.replace(" hours", "")
    ans = ans.replace(" hour", "").replace(" cm", "")
    ans = ans.replace(" km", "")
    return ans

def pad_with_max_val(gt_list, val):
    """if the number of elements in gt is less than MAX_DECODE_STEPS, we pad it with the max value in a class"""
    if len(gt_list) < MAX_DECODE_STEPS:
        gt_list = (
            gt_list
            + (
                np.ones(
                    MAX_DECODE_STEPS - len(gt_list),
                )
                * val
            ).tolist()
        )
    return gt_list

def get_val(qinfo, ans_opt, num_classes_per_puzzle, icon_class_ids, is_one_of_option=False):
    """get the value of the answer option. This code also encodes the value into a number by removing extreneous strings"""
    """ is_one_of_option is True, when ans_opt is one of the options, need not be the correct answer option."""
    where = lambda x, y: np.where(np.array(x) == y)[0][0]
    pid = int(qinfo["puzzle_id"])
    if pid in SEQ_PUZZLES:
        ans = qinfo[ans_opt]
        if pid == 16:
            ans_opt_val = [int(ii) for ii in ans.replace("and", ",").replace(", ,", ",").replace(" ", "").split(",")]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        elif pid == 18:
            ans_opt_val = [int(ii) for ii in ans.split("-")]
            ans_opt_val = pad_with_max_val(ans_opt_val, 5)
        elif pid == 35:
            ans_opt_val = [
                ord(ii) - ord("A") for ii in ans.replace("and", ",").replace(", ,", ",").replace(" ", "").split(",")
            ]
            ans_opt_val = pad_with_max_val(ans_opt_val, 5)
        elif pid == 39:
            ans_opt_val = [ord(ii) - ord("A") for ii in list(ans)]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        elif pid == 63:
            ans_opt_val = [
                int(ii)
                for ii in ans.replace("and", ",")
                .replace("or", ",")
                .replace(", ,", ",")
                .replace("only", "")
                .replace(" ", "")
                .split(",")
            ]
            key = str(63)
            if key in num_classes_per_puzzle:
                ans_opt_val = pad_with_max_val(ans_opt_val, num_classes_per_puzzle[key] - 1)
        elif pid == 100:
            ans_opt_val = [ord(ii) - ord("A") for ii in list(ans)]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        ans_opt_val = np.array(ans_opt_val)

    elif pid == 58:
        # puzzle 58 has answers as <operator><one digit number>, e.g./4,-5, etc.
        # we use +=1, -=2, x=3, /=4. so /4 will be 44, -5=25, +2= 2.
        ans_opt_val = qinfo[ans_opt]
        ans_opt_val = (where(SIGNS, ans_opt_val[0]) + 1) * 10 + int(ans_opt_val[1:])
    elif pid == 25:
        # we need to fix the time in AM/PM format properly.
        ans = qinfo[ans_opt]
        ans_opt_val = int(ans.replace(":00 AM", "").replace(":00 PM", ""))
        if ans.find("PM") > -1:
            ans_opt_val += 12
    else:
        try:
            ans_opt_val = int(qinfo[ans_opt])
        except:
            if len(qinfo[ans_opt]) > 0:
                try:
                    ans_opt_val = ord(qinfo[ans_opt]) - ord("A")
                except:
                    try:
                        ans_opt_val = str_replace(qinfo[ans_opt])
                        ans_opt_val = ans_opt_val.replace("Impossible", "0")  # puzzle 58.
                        if int(qinfo["puzzle_id"]) == 1:  # if the puzzle id is 1, then the options are icon classes.
                            ans_opt_val = "_".join(ans_opt_val.split(" "))
                            if ans_opt_val in icon_class_ids:
                                ans_opt_val = where(icon_class_ids, ans_opt_val)
                            elif ans_opt_val + "s" in icon_class_ids:
                                ans_opt_val = where(icon_class_ids, ans_opt_val + "s")
                        ans_opt_val = int(ans_opt_val)
                    except:
                        print(qinfo)
                        pdb.set_trace()
            else:
                ans_opt_val = ord(ans_opt) - ord("A")
    if not is_one_of_option:  # implies we are encoding the correct answer.
        qinfo["AnswerValue"] = ans_opt_val
    return ans_opt_val


# In[5]:


def split_data(info, split):
    """
    split_type=standard is to use the split_ratio in the instance order
    split_type=exclude is to exclude answers from the split, e.g., train on all answers except say 1, and test 1
    split_type=puzzle is to split the puzzles into the respective ratios. so we don't have to do anything here.
    """
    split_ratio = "80:5:15"
    splits = np.array([int(spl) for spl in split_ratio.split(":")]).cumsum()
    n = len(info)
    if split == "train":
        st = 0
        en = int(np.floor(n * splits[0] / 100.0))
        info = info[st:en]
    elif split == "val":
        st = int(np.ceil(n * splits[0] / 100.0))
        en = int(np.floor(n * splits[1] / 100.0))
        info = info[st:en]
    else:
        st = int(np.ceil(n * splits[1] / 100.0))
        info = info[st:]
    return info


# In[6]:


import random
random.seed(1007)
from torch.utils.data import Dataset, DataLoader

PS_VAL_IDX = [7, 43, 64]
PS_TEST_IDX = [94, 95, 96, 97, 98, 99, 101, 61, 62, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77]
PUZZLE_TYPES = ["counting", "math", "logic", "path", "algebra", "spatial", "pattern", "measure", "order"]

def str_replace_(info, ans_opt):
    ans = info[ans_opt]
    ans = ans.replace(" hours", "")
    ans = ans.replace(" hour", "").replace(" cm", "")
    ans = ans.replace(" km", "")
    ans = ans.replace("Impossible", "0")
    info[ans_opt] = ans
    return ans

class SMARTData(Dataset):
    def __init__(self, split):
        super(SMARTData, self).__init__()
        MAX_VAL = 0
        self.qa_info = []
        self.icon_class_ids = get_icon_dataset_classes(DATASET_DIR + "icon-classes.txt")

        if split == "train":
            puzzle_ids = os.listdir(DATASET_DIR)
            puzzle_ids = np.array(puzzle_ids)[np.array([x.find(".") == -1 for x in puzzle_ids])]
            puzzle_ids = puzzle_ids.tolist()
            val_test = PS_VAL_IDX + PS_TEST_IDX
            val_test = set([str(ii) for ii in val_test])
            puzzle_ids = list(set(puzzle_ids).difference(val_test))
        elif split == "val":
            puzzle_ids = [str(ii) for ii in PS_VAL_IDX]
        else:
            puzzle_ids = [str(ii) for ii in PS_TEST_IDX]

        self.split = split
        self.num_classes_per_puzzle = get_puzzle_class_info(puzzle_ids, self.icon_class_ids)
        print("number of train puzzles = %d" % (len(puzzle_ids)))
        self.transform = Compose(
            [
                Resize(224),  # if the images are of higher resolution. we work with pre-resized 224x224 images.
                # RandomCrop(224),
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
            ]
        )
        self.puzzle_info = {}
        with open(DATASET_DIR + "puzzle_type_info.csv", 'r') as openfile:
            reader = csv.DictReader(openfile)
            for row in reader:
                puzzle_id = row['puzzle_id']
                puzzle_type = row['type']
                self.puzzle_info[puzzle_id] = PUZZLE_TYPES.index(puzzle_type)
        for puzzle_id in puzzle_ids:
          csv_file = "puzzle_%s.csv" % (puzzle_id)
          tqa_info = read_csv(os.path.join(DATASET_DIR, puzzle_id, csv_file), puzzle_id)
          for t in range(len(tqa_info)):
              tqa_info[t]["AnswerValue"] = get_val(tqa_info[t], tqa_info[t]["Answer"], self.num_classes_per_puzzle, self.icon_class_ids)
          self.qa_info += split_data(tqa_info, split)
        print(len(self.qa_info))

    def __len__(self):
        return len(self.qa_info)

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = info["puzzle_id"] + "/"
        im = Image.open(os.path.join(DATASET_DIR, puzzle_root, "img", info["image"])).convert("RGB")
        label = self.puzzle_info[info["puzzle_id"]]
        return self.transform(im), label


# In[10]:


# train_data = SMARTData("train")
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# val_data = SMARTData("val")
# val_loader = DataLoader(val_data, batch_size=64)
test_data = SMARTData("test")
test_loader = DataLoader(test_data, batch_size=64)


# In[11]:

to_train = False
num_classes = 9

import torchvision.models as models
class CustomResNet50(torch.nn.Module):
    def __init__(self, dropout_rate=0.3, num_classes=9):
        super(CustomResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        self.dropout = torch.nn.Dropout(p=dropout_rate) 
        self.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes) 
        self.resnet.fc = torch.nn.Sequential(
            self.dropout,
            self.fc
        )

    def forward(self, x):
        return self.resnet(x)

# Initialize the model with a dropout rate of 0.3
resnet = CustomResNet50(dropout_rate=0.3, num_classes=9)
print(resnet)

 # In[12]:
 
def class_accuracies(y_true, y_pred, classes):
    # Ensure input is numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Dictionary to store accuracy for each class
    accuracies = {}

    # Iterate through each class by index
    for class_index, class_name in enumerate(classes):
        # Get indices where y_true matches the current class index
        class_indices = np.where(y_true == class_index)[0]
        
        # Calculate correct predictions for this class
        correct_predictions = np.sum(y_true[class_indices] == y_pred[class_indices])
        
        # Calculate total predictions for this class
        total_predictions = len(class_indices)
        
        # Calculate accuracy for this class
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
        else:
            accuracy = 0.0  # No predictions for this class
        
        # Store accuracy in the dictionary
        accuracies[class_name] = accuracy

    return accuracies

print(to_train)
if to_train:
    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

    # Define hyperparameters
    learning_rate = 0.001
    num_epochs = 10

    # Initialize ResNet50 model
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)
    model = resnet.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        running_eval_loss = 0.0
        correct_eval_predictions = 0
        total_eval_samples = 0

        for i, batch in enumerate(tqdm(train_loader)):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy and loss
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            running_loss += loss.item()
            
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        torch.save(model.state_dict(), f"resnet_smart_types_{epoch+1}.ckpt")

        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(images)
            loss = criterion(outputs, labels)

            # Track the accuracy and loss
            _, predicted = torch.max(outputs, 1)
            correct_eval_predictions += (predicted == labels).sum().item()
            total_eval_samples += labels.size(0)
            running_eval_loss += loss.item()

        epoch_loss = running_eval_loss / len(val_loader)
        epoch_accuracy = correct_eval_predictions / total_eval_samples

        val_losses.append(epoch_loss)
        val_accuracies.append(epoch_accuracy)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.4f}')
else:
    from sklearn.metrics import classification_report
    # resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    # num_features = resnet.fc.in_features
    # resnet.fc = nn.Linear(num_features, num_classes)
    checkpoint = torch.load('resnet_smart_types_1.ckpt')
    resnet.load_state_dict(checkpoint)
    model = resnet.to(device)
    correct_test_predictions = 0
    total_test_samples = 0
    y_pred = []
    y_true = []
    print("Starting predictions")
    print(len(test_loader))
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(images)

        # Track the accuracy and loss
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu())
        y_true.extend(labels.cpu())
        correct_test_predictions += (predicted == labels).sum().item()
        total_test_samples += labels.size(0)

    epoch_accuracy = correct_test_predictions / total_test_samples
    
    print(f'Test Accuracy: {epoch_accuracy:.4f}')
    accuracies = class_accuracies(y_true, y_pred, PUZZLE_TYPES)
    for k, v in accuracies.items():
        print(k, v)

# In[ ]:




