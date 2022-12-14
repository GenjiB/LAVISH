
for i in {1..8}
do
	echo "run $i times"
	wandb agent genjib/ada_av/k09fgh8o &
	sleep 1m
done

