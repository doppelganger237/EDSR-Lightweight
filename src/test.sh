#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --scale 2 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --save_results

# Depthwise Separable Conv
# python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_dwconv_x2 --reset --epochs 20 --n_threads 0

# Test with Depthwise Separable Conv
#python main.py --data_test Set5 --scale 2 --pre_train ../experiment/edsr_dwconv_x2/model/model_best.pt --test_only --save_results

# fine-tune
#python main.py --model EDSR_DWCONV --scale 2 --patch_size 96 --save edsr_dwconv_x2_ft --reset --epochs 20 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --data_range 1-100/101-105 --n_threads 0

# test fine-tune
#python main.py --data_test Set5+Set14 --scale 2 --pre_train ../experiment/edsr_dwconv_x2_ft/model/model_best.pt --test_only --self_ensemble --save edsr_dwconv_x2_ft_bench --save_results --n_threads 0

# test esdr
#python main.py --data_test Set5+Set14 --scale 2 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --self_ensemble --save edsr_x2_bench --save_results --n_threads 0


# train edsr attention
#python main.py --model EDSR_ATTENTION --scale 2 --epochs 50 --save edsr_attention_x2


# train edsr attention fine-tune
#python main.py --model EDSR_ATTENTION --scale 2 --patch_size 96 --save edsr_attention_x2_ft --reset --epochs 20 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --data_range 1-100/101-105 --n_threads 0

# train dwconv + attention
# python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save edsr_dwconv_attention_x2_ft --reset --epochs 20 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --data_range 1-100/101-105 --n_threads 0 --use_dwconv --use_ca



# -------------------------------
# 训练部分（fine-tune 20 epochs）
# -------------------------------

# 1. EDSR Baseline
# python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2_ft --reset --epochs 20 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --data_range 1-100/101-105 --n_threads 0

# 2. EDSR + DWConv
# python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save edsr_dwconv_x2_ft --reset --epochs 20 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --data_range 1-100/101-105 --n_threads 0 --use_dwconv

# 3. EDSR + Attention
# python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save edsr_attention_x2_ft --reset --epochs 20 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --data_range 1-100/101-105 --n_threads 0 --use_ca

# 4. EDSR + DWConv + Attention
#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save edsr_dwconv_attention_x2_ft --reset --epochs 20 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --data_range 1-100/101-105 --n_threads 0 --use_dwconv --use_ca --use_sa


# -------------------------------
# 验证部分（Set5 + Set14）
# -------------------------------

# 1. Baseline 验证
# python main.py --data_test Set5+Set14 --scale 2 --pre_train ../experiment/edsr_baseline_x2_ft/model/model_best.pt --test_only --self_ensemble --save edsr_baseline_x2_bench --save_results --n_threads 0

# 2. DWConv 验证
# python main.py --data_test Set5+Set14 --scale 2 --pre_train ../experiment/edsr_dwconv_x2_ft/model/model_best.pt --test_only --self_ensemble --save edsr_dwconv_x2_bench --save_results --n_threads 0

# 3. Attention 验证
# python main.py --data_test Set5+Set14 --scale 2 --pre_train ../experiment/edsr_attention_x2_ft/model/model_best.pt --test_only --self_ensemble --save edsr_attention_x2_bench --save_results --n_threads 0

# 4. DWConv + Attention 验证
#python main.py --data_test Set5+Set14 --scale 2 --pre_train ../experiment/edsr_dwconv_attention_x2_ft/model/model_best.pt --test_only --self_ensemble --save edsr_dwconv_attention_x2_bench --save_results --n_threads 0

