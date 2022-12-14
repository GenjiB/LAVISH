
for i in {1..8}
do
	echo "run $i times"
	wandb agent genjib/avqa/ox48orxe &
	sleep 1m
done

