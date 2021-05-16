# PortraitNet论文复现

AI Studio 项目连接：https://aistudio.baidu.com/aistudio/projectdetail/1759729

- EG1800数据集
  - 原论文精度为：96.62%
  - 我达到的精度为：96.634%
- Supervisely_face数据集
  - 原论文精度为：93.43%
  - 我达到的精度为：92.216%
- 运行说明：
  - 1.运行完一个数据集后，需要重启执行器后再去运行第二个数据集的代码，运行环境为32GB
  - 2.两个数据集的训练VisuDL日志分别保存在work/log_eg1800中和work/log_supervisely下。注意eg1800中我是中断运行后又恢复训练，所以有两个.log文件
  - 3.训练一次就评估一次，评估结果保存在work/eg1800_log.txt和work/supervisely_log.txt中（原论文作者也是这样做的）
  - 4.eg1800数据集总共迭代1800epoch，运行到第1660epoch时候，评估效果最优。supervisely数据集总共迭代到1300epoch，运行到1271时候，评估效果最优（第二个数据集由于算力卡限额，就跑到这里了，正常是设置到2000epoch）。
  - 5.训练集和测试集的文件，数据集，评价指标完全都是用原论文作者提供的。
  - 6.我在复现过程中，就增加了图像的入网分辨率，以及一点数据增强上的改动。