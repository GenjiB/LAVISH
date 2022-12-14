import os
import pandas as pd

def download(set, name, t_seg):
    #label = label.replace(" ", "_")  # avoid space in folder name
    path_data = os.path.join(set, "video")
    print(path_data)
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    link_prefix = "https://www.youtube.com/watch?v="

    filename_full_video = os.path.join(path_data, name) + "_full_video.mp4"
    filename = os.path.join(path_data, name) + ".mp4"
    link = link_prefix + name

    if os.path.exists(filename):
        print("already exists, skip")
        return

    print( "download the whole video for: [%s] - [%s]" % (set, name))
    command1 = 'youtube-dl --ignore-config '
    command1 += link + " "
    command1 += "-o " + filename_full_video + " "
    command1 += "-f best "

    #command1 += '-q '  # print no log
    #print command1
    os.system(command1)

    t_start, t_end = t_seg
    t_dur = t_end - t_start
    print("trim the video to [%.1f-%.1f]" % (t_start, t_end))
    command2 = 'ffmpeg '
    command2 += '-ss '
    command2 += str(t_start) + ' '
    command2 += '-i '
    command2 += filename_full_video + ' '
    command2 += '-t '
    command2 += str(t_dur) + ' '
    command2 += '-vcodec libx264 '
    command2 += '-acodec aac -strict -2 '
    command2 += filename + ' '
    command2 += '-y '  # overwrite without asking
    command2 += '-loglevel -8 '  # print no log
    #print(command2)
    os.system(command2)
    try:
        os.remove(filename_full_video)
    except:
        return

    print ("finish the video as: " + filename)


##%% read the label encoding
# filename = "../doc/class_labels_indices.csv"
# lines = [x.strip() for x in open(filename, 'r')][1:]
# label_encode = {}
# for l in lines:
#    l = l[l.find(",")+1:]
#    encode = l.split(",")[0]
#    label_encode[ l[len(encode)+2:-1] ] = encode
#
#
#

# %% read the video trim time indices
filename_source = "data/AVVP_dataset_full.csv"  #
set = "data/LLP_dataset"
df = pd.read_csv(filename_source, header=0, sep='\t')
filenames = df["filename"]
length = len(filenames)
print(length)
names = []
segments = {}
for i in range(length):
    row = df.loc[i, :]
    name = row[0][:11]
    steps = row[0][11:].split("_")
    t_start = float(steps[1])
    t_end = t_start + 10
    segments[name] = (t_start, t_end)
    download(set, name, segments[name])
    names.append(name)
print(len(segments))

