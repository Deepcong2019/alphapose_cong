import os
import re

videos_folder = "D:\\st-gcn\\data\\record_night\\action_videos"
action_folders = os.listdir(videos_folder)
for i in action_folders:
    folders_path = os.path.join(videos_folder, i)
    filenames = os.listdir(folders_path)
    for name in filenames:
        newname = re.split(r'.mkv', name)[0] + '_d1' + '.mkv'
        name_path = os.path.join(folders_path, name)
        newname_path = os.path.join(folders_path, newname)
        os.rename(name_path, newname_path)


