
# export CUDA_VISIBLE_DEVICES=0

#  python3 main_trans.py --epochs=100 --audio_dir=./feats/vggish/ --audio_enc=1 --aug_type=ada --batch_size=4 --decay=0.95 --early_stop=5 \
#  --decay_epoch=1 --delta=0 --forward_dim=512 --gamma=1 --is_a_ori=0 --is_v_ori=0 --qkv_fusion=1 --start_tune_layers=0 --start_fusion_layers=0 \
#  --mode=train --num_head=1 --num_layer=1 --occ_dim=128 --prob_drop=0.4 --prob_drop_occ=0.25 --init_epoch=1 \
#  --st_dir=./feats/r2plus1d_18/ --tsne=0 --video_dir=./feats/res152/   \
#  --Adapter_downsample=2 --num_workers=8 --audio_length=1 --is_audio_adapter_p1=1  --lr=1e-05 --lr_mlp=5e-06 \
#  --alpha=0.5 --beta=0 --wandb=0 --model_name=CLSadd_best_fusionAdapter_reduceX2_G2 #



# for VAR in  0.2
# do

#      python3 main_trans.py --epochs=15 --audio_dir=./feats/vggish/ --audio_enc=1 --aug_type=ada --augment=0 --batch_size=4 --decay=0.95 \
#     --decay_epoch=1 --delta=0 --forward_dim=512 --gamma=1 --is_a_ori=0 --is_v_ori=0 --lr=0.000001 \
#     --mode=train --num_head=1 --num_layer=1 --occ_dim=128 --prob_drop=0.4 --prob_drop_occ=0.25 --init_epoch=1 \
#     --st_dir=./feats/r2plus1d_18/ --tsne=0 --video_dir=./feats/res152/ --vis_smoothing=0.9  --num_workers=8 --audio_length=1 --alpha=0 --beta=$VAR  --wandb=1 --model_name=finetune_disAV${VAR}_lr-6-4
        
# done

# python3 main_trans.py --Adapter_downsample=8 --audio_folder=/data/yanbo/Dataset/AVE_Dataset/raw_audio --batch_size=1 \
# --early_stop=5 --epochs=50 --is_audio_adapter_p1=1 --is_audio_adapter_p2=1 --is_audio_adapter_p3=0 --is_before_layernorm=1 \
# --is_bn=1 --is_fusion_before=1 --is_gate=1 --is_post_layernorm=1 --is_vit_ln=0 --lr=5e-05 --lr_mlp=4e-06 \
# --mode=train --model=MMIL_Net --num_conv_group=2 \
# --num_tokens=2 --num_workers=16 --video_folder=/data/yanbo/Dataset/AVE_Dataset/video_frames \
# --wandb=0 --is_multimodal=1 --model_name=AVISH_V2A


python3 main_trans.py --Adapter_downsample=8 --audio_folder=/data/yanbo/Dataset/AVE_Dataset/raw_audio --batch_size=2 \
--early_stop=5 --epochs=100 --is_audio_adapter_p1=1 --is_audio_adapter_p2=1 --is_audio_adapter_p3=0 --is_before_layernorm=1 \
--is_bn=1 --is_fusion_before=1 --is_gate=1 --is_post_layernorm=1 --is_vit_ln=0 --lr=5e-05 --lr_mlp=4e-06 \
--mode=train --model=MMIL_Net --num_conv_group=2 \
--num_tokens=2 --num_workers=16 --video_folder=/data/yanbo/Dataset/AVE_Dataset/video_frames \
--wandb=1 --is_multimodal=1 --vis_encoder_type=vit --model_name=fixed_av_vit_Tiny16

