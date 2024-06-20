'''
多个文件夹下的多个视频通过alphapose拿到的skeleton结果转化为kinetics-skeleton的json文件格式
'''
import json
import os.path
import re

alphapose_json_folder = 'D:\\AlphaPose\\examples\\res'
out_kinetics_folder = 'D:\\st-gcn\\data\\KTH\\kinetics-skeleton\\kinetics_train'

label_names = ['一道', '七道', '三道', '九道', '二道', '五道', '八道', '六道', '四道']
for jsonfile in os.listdir(alphapose_json_folder):
    alphapose_json_path = os.path.join(alphapose_json_folder, jsonfile)
    with open(alphapose_json_path, 'r') as file:
        alphapose_data = json.load(file)

    action_name = re.split(r'_', jsonfile)[1]
    label_index = label_names.index(action_name)
    # 开始转化
    alphapose_dict = {'data': [{'frame_index': 0, 'skeleton': []}], 'label': action_name, 'label_index': label_index}

    for i in range(len(alphapose_data)):
        dict_in_data = {'frame_index': 0, 'skeleton': []}
        dict_in_skeleton = dict(pose=[], score=[])
        frame_index = int(re.split(r'.jpg', alphapose_data[i]['image_id'])[0]) + 1
        # 26个关节点，前2个x,y,第三个score
        for j in range(26 * 3):
            if (j + 1) % 3 != 0:
                dict_in_skeleton['pose'].append(alphapose_data[i]['keypoints'][j])
            else:
                dict_in_skeleton['score'].append(alphapose_data[i]['keypoints'][j])
        if frame_index != alphapose_dict['data'][-1]['frame_index']:
            dict_in_data['frame_index'] = frame_index
            dict_in_data['skeleton'].append(dict_in_skeleton)
            alphapose_dict['data'].append(dict_in_data)
        else:
            alphapose_dict['data'][-1]['skeleton'].append(dict_in_skeleton)

    alphapose_dict['data'].pop(0)

    with open(os.path.join(out_kinetics_folder, jsonfile), "w") as json_file:
        json.dump(alphapose_dict, json_file)
