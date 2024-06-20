# 制作STGCN需要的kinetis-skeleton数据集,具体如下：<br>
1、python gen_pose_all.py(examples/vis确保为空), 基于alphapose格式的骨骼关节点数据，存放在D:\\AlphaPose\\examples\\res;<br>
2、python gen_json_all.py, 把所有json转为kinetics-skeleton格式的json,结果放在D:\st-gcn\data\KTH\kinetics_skeleton\kinetics_train;<br>
3、python dataset_split.py,7/3切分数据集，自动移动到D:\st-gcn\data\KTH\kinetics_skeleton\kinetics_val;<br>
4、python gen_label_json, 制作kinetics_train_lable.json, kinetics_val_lable.json;<br>