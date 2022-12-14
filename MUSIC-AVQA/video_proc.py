"""
Used to compress video in: https://github.com/ArrowLuo/CLIP4Clip
Author: ArrowLuo
"""
import os
import argparse
import ffmpeg
import subprocess
import time
import multiprocessing
from multiprocessing import Pool
import shutil
try:
	from psutil import cpu_count
except:
	from multiprocessing import cpu_count
# multiprocessing.freeze_support()
from ipdb import set_trace
def compress(paras):
	input_video_path, output_video_path = paras
	output_video_path_ori = output_video_path #.split('.')[0]+'.mp4'
	

	output_video_path = os.path.splitext(output_video_path)[0] #+'.mp4'
	

	## for audio/images extractation
	output_img_path = output_video_path + '/%04d.jpg'
	output_audio_path = output_video_path + '/%04d.wav'
	# output_audio2_path = output_video_path.split('.')[0]+'.wav'
	output_audio2_path = output_video_path + '.wav'
	try:
		# command = ['ffmpeg',
		#     '-y',  # (optional) overwrite output file if it exists
		#     '-i', input_video_path,
		#     '-c:v',
		#     'libx264',
		#     '-c:a',
		#     'libmp3lame',
		#     '-b:a',
		#     '128K',
		#     '-max_muxing_queue_size', '9999',
		# 	'-vf',
		#     'fps=3 ',  # scale to 224 "scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'"
		#     # '-max_muxing_queue_size', '9999',
		# #    "scale=224:224",
		# #    '-c:a', 'copy',
		# #    'fps=fps=30',  # frames per second
		#     output_video_path_ori,
		#     ]

		### ori compressed ---------->
		# command = ['ffmpeg',
        #            '-y',  # (optional) overwrite output file if it exists
        #            '-i', input_video_path,
        #            '-filter:v',
        #            'scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'',  # scale to 224
        #            '-map', '0:v',
        #            '-r', '3',  # frames per second
        #            output_video_path_ori,
        #            ]
		########### <----------------
	
		############# for extract images ##############
		# command = ['ffmpeg',
		# 	'-y',  # (optional) overwrite output file if it exists
		# 	'-i', input_video_path,
		# 	'-vf',
		# 	'fps=1',
		# 	output_img_path,
		# ]
		########## end extract images ###################################


		######### for extract audio ----->
		# ffmpeg -i /playpen-iop/yblin/v1-2/train/v_XazKuBawFCM.mp4 -map 0:a -f segment -segment_time 10 -acodec pcm_s16le -ac 1 -ar 16000 /playpen-iop/yblin/v1-2/train_audio/output_%03d.wav
		# ffmpeg -y -i /playpen-iop/yblin/yk2/raw_videos_all/low_all_val/EpNUSTO2BI4.mp4 -map 0:a -f segment -segment_time 10000000 -acodec pcm_s16le -ac 1 -ar 16000 /playpen-iop/yblin/yk2/audio_raw_val/EpNUSTO2BI4.wav
		command = ['ffmpeg',
			'-y',  # (optional) overwrite output file if it exists
			'-i', input_video_path,
			'-acodec', 'pcm_s16le', '-ac', '1',
			'-ar', '16000', # resample 
			output_audio2_path,
		]
		### <------
		

		######### for extract audio 2 ###########
		# command = ['ffmpeg',
		# 	'-y',  # (optional) overwrite output file if it exists
		# 	'-i', input_video_path,
		# 	'-map','0:a', '-f', 'segment',
		# 	'-segment_time', '10000000', # seconds here
		# 	'-acodec', 'pcm_s16le', '-ac', '1',
		# 	'-ar', '16000', # resample 
		# 	output_audio_path,
		# ]
		#####################

		# command= [
		# 	'mkdir',
		# 	output_video_path,  # (optional) overwrite output file if it exists
		# ]

		print(command)
	
		# ffmpeg -y -i /playpen-iop/yblin/v1-2/val/v_6VT2jBflMAM.mp4 -vf  fps=3 /playpen-iop/yblin/v1-2/val_low_scale/v_6VT2jBflMAM.mp4
		ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = ffmpeg.communicate()
		retcode = ffmpeg.poll()
		# print something above for debug
	except Exception as e:
		raise e

def prepare_input_output_pairs(input_root, output_root):
	input_video_path_list = []
	output_video_path_list = []
	for root, dirs, files in os.walk(input_root):
		
		for file_name in files:
			input_video_path = os.path.join(root, file_name)
			
			output_video_path = os.path.join(output_root, file_name)
			if os.path.exists(output_video_path):
				pass
			else:
				input_video_path_list.append(input_video_path)
				output_video_path_list.append(output_video_path)

	return input_video_path_list, output_video_path_list
# ffmpeg -y -i /nas/longleaf/home/yanbo/dataset/msvd_data/MSVD_Videos/-4wsuPCjDBc_5_15.avi -vf "fps=3,scale=224:224" /nas/longleaf/home/yanbo/dataset/msvd_data/MSVD_Comp2/-4wsuPCjDBc_5_15.avi 


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Compress video for speed-up')
	parser.add_argument('--input_root', type=str, help='input root')
	parser.add_argument('--output_root', type=str, help='output root')
	args = parser.parse_args()

	input_root = args.input_root
	output_root = args.output_root

	assert input_root != output_root

	if not os.path.exists(output_root):
		os.makedirs(output_root, exist_ok=True)

	input_video_path_list, output_video_path_list = prepare_input_output_pairs(input_root, output_root)


	print("Total video need to process: {}".format(len(input_video_path_list)))
	num_works = cpu_count()
	print("Begin with {}-core logical processor.".format(num_works))


	# pool = Pool(num_works)
	pool = Pool(128)

	data_dict_list = pool.map(compress,
							  [(input_video_path, output_video_path) for
							   input_video_path, output_video_path in
							   zip(input_video_path_list, output_video_path_list)])
	pool.close()
	pool.join()

