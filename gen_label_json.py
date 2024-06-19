import os
import re
import json


# 读取kinetics-skeleton的json文件格式,查看true和True是否有区别
kinetics_json = 'D:\\st-gcn\\data\\Kinetics\\kinetics-skeleton\\kinetics_val_label.json'
with open(kinetics_json, 'r') as file:
    kinetics_data = json.load(file)


dataset_type = ['train', 'val']
label_names = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
for name in dataset_type:
    output_json = 'D:\\st-gcn\\data\\KTH\\kinetics-skeleton\\kinetics_{}_label.json'.format(name)
    kinetics_path = 'D:\\st-gcn\\data\\KTH\\kinetics-skeleton\\kinetics_{}'.format(name)
    label_dict = {}
    for jsonfile in os.listdir(kinetics_path):
        json_name = jsonfile.split('.json')[0]
        label = re.split(r'_', jsonfile)[1]
        label_index = label_names.index(label)
        label_dict[json_name] = {"has_skeleton": True, "label": label, "label_index": label_index}
    with open(output_json, 'w') as file:
        json.dump(label_dict, file)





