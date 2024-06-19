'''
单个视频通过alphapose拿到的skeleton结果转化为kinetics-skeleton的json文件格式
'''
import json
import re
# 读取kinetics-skeleton的json文件格式
kinetics_json = 'D:\\st-gcn\\data\\Kinetics\\kinetics-skeleton\\kinetics_val\\_77cew2otmY.json'
with open(kinetics_json, 'r') as file:
    kinetics_data = json.load(file)


alphapose_json = 'D:\\AlphaPose\\examples\\res\\person01_jogging_d1_uncomp(003320-021320).avi.json'
# alphapose_json = 'D:\\AlphaPose\\examples\\res\\ntu_sample.avi.json'
with open(alphapose_json, 'r') as file:
    alphapose_data = json.load(file)


# 开始转化
alphapose_dict = {'data':[{'frame_index': 0, 'skeleton': []}], 'label': 's', 'label_index': 0}

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

with open("output.json", "w") as json_file:
    json.dump(alphapose_dict, json_file)

