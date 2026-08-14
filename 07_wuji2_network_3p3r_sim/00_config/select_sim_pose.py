"""只修改下面三个整数，然后运行../02_scripts/04_build_selected_sim_3p3r.py。"""

# 09测试中的场景编号，当前允许0、1、2、3、4。
SCENE_INDEX = 0

# 当前场景scene_manifest中的物体分割ID，不是物体数组下标。
OBJECT_SEGMENTATION_ID = 14

# 该物体在balanced碰撞过滤结果中按综合评分降序的名次，从1开始。
# 例如5就是Top35可视化中红色编号“05”的姿态。
FILTERED_RANK = 5
