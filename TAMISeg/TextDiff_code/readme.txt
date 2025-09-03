
textdiff训练

step1: 将第一阶段训练的run_files文件夹里的训练好的权重保存到TextDiff_code的checkpoints文件夹下，目前该文件夹下已经给了一个示例，到时候替换掉就行（名字需要一样）
step2: 训练
python train.py --exp experiments/qata_cov19_v2_2/condseg.json
训练结果会保存在当前文件夹的saved_textdiff文件夹下
