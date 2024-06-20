import os
import re
import random
import shutil

all_json_folder = 'D:\\st-gcn\\data\\KTH\\kinetics-skeleton\\kinetics_train'
val_json_folder = 'D:\\st-gcn\\data\\KTH\\kinetics-skeleton\\kinetics_val'
label_names = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
all_json_path = os.listdir(all_json_folder)
val_json_names = random.sample(all_json_path, int(len(all_json_path) * 0.3))
for path in val_json_names:
     val_json_path = os.path.join(all_json_folder, path)
     shutil.move(val_json_path, val_json_folder)
