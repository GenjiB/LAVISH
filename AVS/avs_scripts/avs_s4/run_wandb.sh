

for i in {1..8}
do
	echo "run $i times"
	wandb agent genjib/ada_av_segmentation/ix3wtgic &
	sleep 1m
done

