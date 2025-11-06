
# 测试 Ghost+
#python main.py --model BFFN --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/bffn_ghost/model/model_best.pt --test_only --self_ensemble --n_threads 0


#python main.py --model BFFN --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/bffn_new/model/model_best.pt --test_only --n_threads 0

# 训练部分


#python main.py --model BFFN --scale 3 --patch_size 96 --save bffn_x3  --pre_train ../experiment/bffn_ghost/model/model_best.pt --n_threads 0 --lr 5e-4 --reset



python main.py --model PFDN --scale 2 --patch_size 64 --save test --n_threads 10 --lr 5e-4 --n_resblocks 6 --n_feats 52 --batch_size 16 --use_amp

#python main.py --model bsrn --scale 2 --patch_size 96 --save rlfn_bsconv --n_threads 0 --batch_size 64 --lr 5e-4

#python main.py --model RepRLFN --scale 2 --patch_size 96 --save test --n_threads 0 --lr 5e-4 --batch_size 64


#python main.py --model SPAN --scale 2 --patch_size 96 --save span --n_threads 0 --lr 1e-4 --batch_size 16


#python main.py --model PFDN --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/pfdn/model/model_best.pt --test_only --n_threads 0 --n_resblocks 6 --n_feats 52



#python main.py --model BFFN --data_test Set5+Set14+B100+Urban100 --scale 2 --n_resblocks 6 --pre_train ../experiment/bffn_res6/model/model_best.pt --test_only --n_threads 0



# Test Baseline

#python main.py --model BFFN --scale 2 --patch_size 128 --save bffn_128 --load bffn_128 --n_threads 0 --lr 5e-4 --n_resblocks 8 --resume -1

# 测试部分

#python main.py --model BFFN --data_test Set5+Set14+B100+Urban100 --scale 2 --n_resblocks 8 --pre_train ../experiment/bffn/model/model_best.pt --test_only --n_threads 0

#python main.py --model BFFN --data_test Set5+Set14+B100+Urban100 --scale 2 --n_resblocks 6 --pre_train ../experiment/bffn_res6/model/model_best.pt --test_only --n_threads 0

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

