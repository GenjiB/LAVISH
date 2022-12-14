
# LAVISH for Audio-visual Segmentation


### 📝 Preparation 
1. `pip3 install requirements.txt`
2. Dataset: [Audio-Visual Segmentation](https://github.com/OpenNLPLab/AVSBench)
3. We follow exact the same setting data format as [AVS](https://github.com/OpenNLPLab/AVSBench).



### 📚 Train and evaluate
```shell
python3 train.py --Adapter_downsample=4 --audio_length=1 --early_stop=5 --is_before_layernorm=1 --is_bn=0 --is_gate=0 --is_multimodal=1 --is_post_layernorm=1 --is_vit_ln=1 --lr=0.0001 --mask_pooling_type=avg --max_epoches=30 --num_conv_group=2 --num_tokens=16 --num_workers=8 --session_name=S4_pvt --tpavi_va_flag=1 --train_batch_size=4 --val_batch_size=1 --visual_backbone=pvt
```




### 🎓 Cite

If you use this code in your research, please cite:

```bibtex
@InProceedings{LAVISH_arxiv2022,
author = {Lin, Yan-Bo an Sung, Yi-Lin and Lei, Jie and Bansal, Mohit and Bertasius, Gedas},
title = {Vision Transformers are Parameter-Efficient Audio-Visual Learners},
booktitle = {arXiv},
year = {2022}
}
```