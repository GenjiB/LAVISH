
# LAVISH for Audio-visual event localization


### 📝 Preparation 
1. `pip3 install requirements.txt`
2. Dataset: [Audio-visual event localization](https://github.com/YapengTian/AVE-ECCV18)
3. extract video frames in 3 fps (You can define ANY fps. The dataloader will only sample 10 frames in total.)
4. extract audio into *.wav


### 💿 Extract images and audio features. (We also provie preprocessed data [here](https://huggingface.co/datasets/genjib/LAVISHData/))
```shell
AVE_Dataset/
├── video_frames/
│       └── VIDEO_NAME/
│           ├── 0001.jpg
│           ├── ...
│           └── 00...jpg
│
└──  raw_audio/
        └── VIDEO_NAME.wav
```




### 📚 Train and evaluate
```shell
python3 main_trans.py --Adapter_downsample=8 --audio_folder=$PATH/raw_audio --batch_size=2 --early_stop=5 --epochs=50 --is_audio_adapter_p1=1 --is_audio_adapter_p2=1 --is_audio_adapter_p3=0 --is_before_layernorm=1 --is_bn=1 --is_fusion_before=1 --is_gate=1 --is_post_layernorm=1 --is_vit_ln=0 --lr=5e-05 --lr_mlp=4e-06 --mode=train --num_conv_group=2 --num_tokens=2 --num_workers=16 --video_folder=$PATH/video_frames  --is_multimodal=1 --vis_encoder_type=swin
```
#### You can also try another version by running `run_v2.sh`




### 🎓 Cite

If you use this code in your research, please cite:

```bibtex
@InProceedings{LAVISH_CVPR2023,
author = {Lin, Yan-Bo and Sung, Yi-Lin and Lei, Jie and Bansal, Mohit and Bertasius, Gedas},
title = {Vision Transformers are Parameter-Efficient Audio-Visual Learners},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2023}
}
```