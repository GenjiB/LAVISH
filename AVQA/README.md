
# LAVISH for Audio-Visual Question Answering


### 📝 Preparation 
1. `pip3 install requirements.txt`
2. Dataset: [MUSIC AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA)
3. We follow exact the same setting data format as [MUSIC AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA).



### 📚 Train and evaluate
```shell
python3 net_grd_avst/main_avst.py --Adapter_downsample=8 --audio_dir=/data/yanbo/Dataset/AVQA/vggish --batch-size=1 --early_stop=5 --epochs=30 --is_before_layernorm=1 --is_bn=0 --is_gate=1 --is_multimodal=1 --is_post_layernorm=1 --is_vit_ln=1 --lr=8e-05 --lr_block=3e-06 --num_conv_group=4 --num_tokens=64 --num_workers=16 --video_res14x14_dir=/data/yanbo/Dataset/AVQA/ --wandb=1
```




### 🎓 Cite

If you use this code in your research, please cite:

```bibtex
@InProceedings{LAVISH_CVPR2023,
author = {Lin, Yan-Bo an Sung, Yi-Lin and Lei, Jie and Bansal, Mohit and Bertasius, Gedas},
title = {Vision Transformers are Parameter-Efficient Audio-Visual Learners},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2023}
}
```