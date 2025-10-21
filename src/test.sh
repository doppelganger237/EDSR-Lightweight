#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_attention_x2_full --reset --epochs 300 --n_threads 0 --use_dwconv --use_attention --save_results 

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_x2 --load lumisr_dwconv_x2 --pre_train ../experiment/lumisr_dwconv_x2/model/model_latest.pt --epochs 300 --n_threads 0 --use_dwconv --save_results 

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_attention_x2 --epochs 300 --n_threads 0 --use_attention --save_results 
#python main.py --model EDSR_VARIANTS --scale 2 --pre_train ../experiment/lumisr_dwconv_attention_x2_full/model/model_best.pt --test_only --use_dwconv --use_attention --n_colors 1

# Train DW_only
#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 64  --n_resblocks 8 --save sr_dw_x2 --reset --epochs 300 --n_threads 0 --use_dwconv --save_results

python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96  --save sr_att --epochs 300 --n_threads 0 --use_attention --save_results



# Train Baseline
#python main.py --model EDSR --scale 2 --save EDSR_Baseline_x2 --epochs 300 --n_threads 0 --save_results

# Test Baseline
# Train Att_only
#python main.py --model EDSR_VARIANTS --scale 2  --patch_size 96 --batch_size 24 --save sr_attention_x2 --epochs 300 --n_threads 0 --use_attention --save_results

#python main.py --model EDSR_VARIANTS --scale 2  --patch_size 96 --batch_size 24 --save sr_dw --epochs 300 --n_threads 0 --use_dwconv --save_results --use_attention

# Test Baseline
#python main.py --model EDSR --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --n_threads 0 --only_y

# 测试 Baseline 与官方数据相同，不用再测试
# [DIV2K x2]      PSNR: 34.609 (Best: 34.609 @epoch 1) | SSIM: 0.9399 (Best: 0.9399 @epoch 1)
#python main.py --model EDSR --data_test DIV2K --data_range 801-900 --scale 2 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --n_threads 0

# 测试 Baseline DIV2K801-810
# [DIV2K x2]      PSNR: 35.640 (Best: 35.640 @epoch 1) | SSIM: 0.9415 (Best: 0.9415 @epoch 1)
#python main.py --model EDSR --data_test DIV2K --data_range 801-810 --scale 2 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --n_threads 0


# Test Att_only 
#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/sr_attention_x2/model/model_best-6.pt --test_only --n_threads 0 --use_attention


# Test Att_only DIV2K100
# [DIV2K x2]      PSNR: 34.413 (Best: 34.413 @epoch 1) | SSIM: 0.9384 (Best: 0.9384 @epoch 1)
#python main.py --model EDSR_VARIANTS --data_test DIV2K --data_range 801-900 --scale 2 --pre_train ../experiment/sr_attention_x2/model/model_best-6.pt --test_only --n_threads 0 --use_attention


#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/sr_attention_x2/model/model_best-7.pt --test_only --self_ensemble --n_threads 0 --use_attention --use_dwconv

#Test Conv
#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/lumisr_dwconv_x2/model/model_best.pt --test_only --self_ensemble --n_threads 0 --use_dwconv

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_x2_from_base --pre_train ../models/edsr_baseline_x2-1bc95232.pt --epochs 300 --n_threads 0 --use_dwconv --save_results

