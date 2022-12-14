import shutil
import subprocess
import os
import argparse
import glob

def extract_frames(video, dst):
    command1 = 'ffmpeg '
    command1 += '-i ' + video + " "
    command1 += '-y' + " "
    command1 += "-r " + "8 "
    command1 += '{0}/%06d.jpg'.format(dst)
    print(command1)
    #    print command1
    os.system(command1)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='data/LLP_dataset/frame')
    parser.add_argument('--video_path', dest='video_path', type=str, default='data/LLP_dataset/video')
    args = parser.parse_args()

    vid_list = os.listdir(args.video_path)

    for vid_id in vid_list:
        name = os.path.join(args.video_path, vid_id)
        dst = os.path.join(args.out_dir, vid_id[:-4])
        print(dst)
        if not os.path.exists(dst):
            os.makedirs(dst)
        extract_frames(name, dst)
        print("finish video id: " + vid_id)
