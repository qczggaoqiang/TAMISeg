import os
import random
import pandas as pd
import shutil
from tqdm import tqdm



def read_xlsx_and_save(xlsx_path, txt_path, tag):
    # 读取所有行，从第二列（索引为1）开始的所有列
    df = pd.read_excel(
        xlsx_path,
    )
    df_filtered_list = df.iloc[1:, :].values[:, 0].tolist()
    print(len(df_filtered_list))
    for i in tqdm(list(range(len(df_filtered_list)))):
        img_name = df_filtered_list[i].split('.')[0]
        img_path = os.path.join('/root/autodl-tmp/new_model_text_prompt/TextDiff_code/datasets/Covid19/frames', df_filtered_list[i])
        label_path = os.path.join('/root/autodl-tmp/new_model_text_prompt/TextDiff_code/datasets/Covid19/masks', df_filtered_list[i])
        shutil.copy(img_path, os.path.join('/root/autodl-tmp/new_model_text_prompt/TextDiff_code/datasets/Covid19', tag, 'frames'))
        shutil.copy(label_path, os.path.join('/root/autodl-tmp/new_model_text_prompt/TextDiff_code/datasets/Covid19', tag, 'masks'))
        with open(txt_path, 'a+') as w:
            w.write(img_name + '\n')
    

train_xlsx_path = '/root/autodl-tmp/new_model_text_prompt/TextDiff_code/Medical_txt_for_segmentation/Train_text_MosMedData.xlsx'
save_train_img_name_txt_path = '/root/autodl-tmp/new_model_text_prompt/TextDiff_code/datasets/Covid19/train.txt'
read_xlsx_and_save(train_xlsx_path, save_train_img_name_txt_path, 'train')
val_xlsx_path = '/root/autodl-tmp/new_model_text_prompt/TextDiff_code/Medical_txt_for_segmentation/Val_text_MosMedData.xlsx'
save_val_img_name_txt_path = '/root/autodl-tmp/new_model_text_prompt/TextDiff_code/datasets/Covid19/val.txt'
read_xlsx_and_save(val_xlsx_path, save_val_img_name_txt_path, 'val')
test_xlsx_path = '/root/autodl-tmp/new_model_text_prompt/TextDiff_code/Medical_txt_for_segmentation/Test_text_MosMedData.xlsx'
save_test_img_name_txt_path = '/root/autodl-tmp/new_model_text_prompt/TextDiff_code/datasets/Covid19/test.txt'
read_xlsx_and_save(test_xlsx_path, save_test_img_name_txt_path, 'test')


    




# img_dir = '/root/autodl-tmp/new_model_text_prompt/TextDiff_code/datasets/Covid19/frames'
# img_list = os.listdir(img_dir)

# random.shuffle(img_list)
# train_len = int(0.85 * len(img_list))


# save_train_img_name_txt_path = '/root/autodl-tmp/new_model_text_prompt/TextDiff_code/datasets/Covid19/train.txt'
# save_val_img_name_txt_path = '/root/autodl-tmp/new_model_text_prompt/TextDiff_code/datasets/Covid19/val.txt'




# for i in range(len(img_list)):
#     img_name = img_list[i].split('.')[0]
#     if i < train_len:
#         with open(save_train_img_name_txt_path, 'a+') as w:
#             w.write(img_name + '\n')
#     else:
#         with open(save_val_img_name_txt_path, 'a+') as w:
#             w.write(img_name + '\n')


