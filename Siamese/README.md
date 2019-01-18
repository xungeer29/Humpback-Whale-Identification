# Siamese
## dataset.py
* 根据image hash去重的pickle结果与train.csv 中对应起来，找到每张图像中鲸鱼的名字，
相当于重新生成新的train.csv，根据这个生成图像id，统计每个id的图像数量，存在一个list中，
使用超过两张的图像生成相同配对，不超过
