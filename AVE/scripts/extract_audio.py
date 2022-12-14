import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip

video_pth =  "data/LLP_dataset/video"
sound_list = os.listdir(video_pth)
save_pth =  "data/LLP_dataset/audio"

for audio_id in sound_list:
    name = os.path.join(video_pth, audio_id)
    audio_name = audio_id[:-4] + '.wav'
    exist_lis = os.listdir(save_pth)
    if audio_name in exist_lis:
        print("already exist!")
        continue
    try:
        video = VideoFileClip(name)
        audio = video.audio
        audio.write_audiofile(os.path.join(save_pth, audio_name), fps=16000)
        print("finish video id: " + audio_name)
    except:
        print("cannot load ", name)