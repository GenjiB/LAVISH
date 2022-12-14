python net_grd_avst/main_avst.py  --mode train \
	--audio_dir /data/yanbo/Dataset/AVQA/vggish \
	--video_res14x14_dir /data/yanbo/Dataset/AVQA/ \
	--wandb 0 --num_workers 32 --batch-size 32 --model_name swinv2_tune_av+vggish