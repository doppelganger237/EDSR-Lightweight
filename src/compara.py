import cv2
import os
import matplotlib.pyplot as plt



def generate_bicubic(lr_path, save_path, scale=4):
    """
    从 LR 图像生成 Bicubic 上采样结果。
    仅在需要时手动调用。
    """
    lr_full_path = os.path.normpath(os.path.join(base_dir, lr_path))
    save_full_path = os.path.normpath(os.path.join(base_dir, save_path))
    if not os.path.exists(lr_full_path):
        print(f"⚠️ 找不到 LR 图片: {lr_full_path}")
        return
    lr = cv2.imread(lr_full_path)
    if lr is None:
        print(f"⚠️ 无法读取 LR 图片: {lr_full_path}")
        return
    h_hr = lr.shape[0] * scale
    w_hr = lr.shape[1] * scale
    bicubic = cv2.resize(lr, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
    os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
    cv2.imwrite(save_full_path, bicubic)
    print(f"✅ 已生成 Bicubic 上采样结果: {save_full_path}")



base_dir = os.path.dirname(os.path.abspath(__file__))




def compare_images(models, dataset_name, crop_coords):
    y, x, h, w = crop_coords
    output_dir = os.path.join(base_dir, f"../compare_results/{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)

    hr_path = os.path.normpath(os.path.join(base_dir, models["HR"]))
    hr_img = cv2.imread(hr_path)[..., ::-1]  # BGR -> RGB
    hr_img = hr_img.astype("uint8").copy()
    cv2.rectangle(hr_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "HR_full.png"), hr_img[..., ::-1])  # RGB -> BGR
    print(f"✅ [{dataset_name}] 已保存全图带标记: HR_full.png")

    cropped_images = {}

    for name, rel_path in models.items():
        path = os.path.normpath(os.path.join(base_dir, rel_path))
        print(f"🔍 [{dataset_name}] 处理图片: {path}")
        if not os.path.exists(path):
            print(f"⚠️ [{dataset_name}] 找不到图片: {path}")
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ [{dataset_name}] 无法读取: {path}")
            continue

        img = img[..., ::-1].copy()  # BGR -> RGB
        crop_h = min(h, img.shape[0] - y)
        crop_w = min(w, img.shape[1] - x)

        crop = img[y:y+crop_h, x:x+crop_w].copy()
        save_crop_path = os.path.join(output_dir, f"{name}_crop.png")
        cv2.imwrite(save_crop_path, crop[..., ::-1])  # RGB -> BGR
        print(f"✅ [{dataset_name}] 已保存裁切区域: {save_crop_path}")

        cropped_images[name] = crop

    # 绘图展示
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(cropped_images) + 1, 1)
    plt.imshow(hr_img)
    plt.title("HR_full")
    plt.axis('off')

    for idx, (name, crop_img) in enumerate(cropped_images.items(), start=2):
        plt.subplot(1, len(cropped_images) + 1, idx)
        plt.imshow(crop_img)
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()




models_set14_ppt3 = {
    "HR": "../dataset/benchmark/Set14/HR/ppt3.png",
    "Bicubic": "../experiment/sr/set14_ppt3_x4_Bicubic.png",
    "VDSR": "../experiment/sr/set14_ppt3_x4_VDSR.png",
    "CARN": "../experiment/sr/set14_ppt3_x4_CARN.png",
    "ShuffleMixer": "../experiment/sr/set14_ppt3_x4_ShuffleMixer.png",
    "PFDN": "../experiment/sr/set14_ppt3_x4_PFDN.png",
}

models_set14_baboon = {
    "HR": "../dataset/benchmark/Set14/HR/baboon.png",
    "Bicubic": "../experiment/sr/set14_baboon_x4_Bicubic.png",
    "VDSR": "../experiment/sr/set14_baboon_x4_VDSR.png",
    "CARN": "../experiment/sr/set14_baboon_x4_CARN.png",
    "ShuffleMixer": "../experiment/sr/set14_baboon_x4_ShuffleMixer.png",
    "PFDN": "../experiment/sr/set14_baboon_x4_PFDN.png",
}

# 调用示例1：使用 Set14 的 models 和裁切坐标

# 调用示例2：定义 Urban100 的 models 并调用
models_urban = {
    "HR": "../dataset/benchmark/Urban100/HR/img067.png",
    "Bicubic": "../experiment/sr/urban100_img067_x4_Bicubic.png",
    "VDSR": "../experiment/sr/urban100_img067_x4_VDSR.png",
    "CARN": "../experiment/sr/urban100_img067_x4_CARN.png",
    "ShuffleMixer": "../experiment/sr/urban100_img067_x4_ShuffleMixer.png",
    "PFDN": "../experiment/pfdn_x4/results-Urban100/img067_x4_SR.png",
}

#generate_bicubic("../dataset/benchmark/Urban100/LR_bicubic/x4/img067x4.png", "../experiment/sr/x4/urban100/img67/img067_x4_Bicubic.png", scale=4)

#generate_bicubic("../dataset/benchmark/Set14/LR_bicubic/x4/ppt3x4.png", "../experiment/sr/set14_ppt3_x4_Bicubic.png", scale=4)

#generate_bicubic("../dataset/benchmark/Set14/LR_bicubic/x4/baboonx4.png", "../experiment/sr/set14_baboon_x4_Bicubic.png", scale=4)

#compare_images(models_set14_ppt3, "Set14_ppt3", (70, 115, 74, 102))


compare_images(models_set14_baboon, "Set14_baboon", (310, 300, 96, 128))

#compare_images(models_urban, "Urban100_img067", (80, 100, 96, 128))
