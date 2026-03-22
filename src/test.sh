# 训练部分

#python main.py --model PFRN --scale 2 --patch_size 128 --n_resblocks 8 --n_feats 58 --save pfrn --n_threads 0 --lr 5e-4 --batch_size 16 --epochs 1000 --decay 200-400-600-800

# python main.py --model PFRN --scale 2 --patch_size 128 --n_resblocks 6 --n_feats 52 --save pfrn --load pfrn --n_threads 0 --lr 5e-4 --batch_size 16 --resume -1 --epochs 1000 --decay 200-400-600-800

python main.py --model PFRN --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/pfrn_x2/model/model_best.pt --test_only --n_threads 0 --n_resblocks 8 --n_feats 52




#python main.py --model PFDN --scale 2 --patch_size 128 --n_resblocks 6 --n_feats 52 --save pfdn --load pfdn --n_threads 0 --lr 5e-4 --batch_size 64 --resume -1 --epochs 1000 --decay 200-400-600-800

#python main.py --model PFDN --scale 3 --patch_size 96 --save test --pre_train ../experiment/pfdn_x2/model/model_best.pt --n_threads 0 --lr 5e-4 --n_resblocks 6 --n_feats 52 --batch_size 16 

#python main.py --model PFDN --data_test Set5+Set14+B100+Urban100 --scale 4 --pre_train ../experiment/pfdn_x4/model/model_best.pt --n_threads 0 --n_resblocks 6 --n_feats 52  --test_only --save_results

#python main.py --model PFDN --data_test DIV2K --data_range 801-900 --scale 4 --pre_train ../experiment/pfdn_x4/model/model_best.pt --n_threads 0 --n_resblocks 6 --n_feats 52  --test_only 

#python main.py --model EDSR --data_test DIV2K --data_range 801-900 --scale 4 --pre_train ../models/edsr_baseline_x4-6b446fab.pt --test_only  --n_resblocks 16 --n_feats 64 --n_threads 0

#python main.py --model PFDN --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/pfdn_x2/model/model_best.pt --test_only --n_threads 0 --n_resblocks 6 --n_feats 52


#python main.py --model PFDN --data_test Set5+Set14+B100+Urban100 --scale 3 --pre_train ../experiment/pfdn_x3/model/model_best.pt --test_only --n_threads 0 --n_resblocks 6 --n_feats 52




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



# 测试 Baseline x4 DIV2K801-810
# [DIV2K x4]      PSNR: 29.577 (Best: 29.577 @epoch 1) | SSIM: 0.8163 (Best: 0.8163 @epoch 1)
#python main.py --model EDSR --data_test DIV2K --data_range 801-810 --scale 3 --pre_train ../models/edsr_baseline_x3-abf2a44e.pt --test_only --n_threads 0  --n_resblocks 16 --n_feats 64 

# Test Att_only 
#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/sr_attention_x2/model/model_best-6.pt --test_only --n_threads 0 --use_attention


# Test Att_only DIV2K100
# [DIV2K x2]      PSNR: 34.413 (Best: 34.413 @epoch 1) | SSIM: 0.9384 (Best: 0.9384 @epoch 1)
#python main.py --model EDSR_VARIANTS --data_test DIV2K --data_range 801-900 --scale 2 --pre_train ../experiment/sr_attention_x2/model/model_best-6.pt --test_only --n_threads 0 --use_attention


#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/sr_attention_x2/model/model_best-7.pt --test_only --self_ensemble --n_threads 0 --use_attention --use_dwconv

#Test Conv
#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/lumisr_dwconv_x2/model/model_best.pt --test_only --self_ensemble --n_threads 0 --use_dwconv

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_x2_from_base --pre_train ../models/edsr_baseline_x2-1bc95232.pt --epochs 300 --n_threads 0 --use_dwconv --save_results

