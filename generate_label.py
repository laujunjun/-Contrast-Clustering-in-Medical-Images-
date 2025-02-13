import os
import pandas as pd

# 读取包含slide_id和label的CSV文件
label_df = pd.read_csv('/path/label_all_slide.csv')

# 创建一个空的DataFrame来存储最终结果
batch_size = 1000  # 设置批次大小
output_list = []

# 遍历父目录中的所有子文件夹
parent_dir = '/path/to/par_dir'
output_csv_path = 'path/label.csv'

for subdir in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir)
    if os.path.isdir(subdir_path):
        # 获取当前子文件夹的slide_id
        slide_id = subdir
        
        # 从label_df中匹配label
        if slide_id in label_df['slide_id'].values:
            label_str = label_df.loc[label_df['slide_id'] == slide_id, 'label'].values[0]
            label = 1 if label_str == 'metastasis' else 0 if label_str == 'no_metastasis' else -1
        else:
            label = -1  # 如果没有找到对应的标签，设置为-1表示未知

        # 遍历子文件夹中的jpg文件
        for file in os.listdir(subdir_path):
            if file.endswith('.jpg'):
                patch_name = file.split('.')[0]  # 去掉扩展名
                output_list.append({'Sample_Name': patch_name, 'label': label})

                # 当达到批次大小时，写入CSV
                if len(output_list) >= batch_size:
                    output_df = pd.DataFrame(output_list)
                    output_df.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)
                    output_list = []  # 重置列表以准备下一批

# 处理剩余数据
if output_list:
    output_df = pd.DataFrame(output_list)
    output_df.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)
