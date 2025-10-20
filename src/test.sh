#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_attention_x2_full --reset --epochs 300 --n_threads 0 --use_dwconv --use_attention --save_results 

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_x2 --load lumisr_dwconv_x2 --pre_train ../experiment/lumisr_dwconv_x2/model/model_latest.pt --epochs 300 --n_threads 0 --use_dwconv --save_results 

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_attention_x2 --epochs 300 --n_threads 0 --use_attention --save_results 
#python main.py --model EDSR_VARIANTS --scale 2 --pre_train ../experiment/lumisr_dwconv_attention_x2_full/model/model_best.pt --test_only --use_dwconv --use_attention --n_colors 1

# Train DW_only
#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 64  --n_resblocks 8 --save sr_dw_x2 --reset --epochs 300 --n_threads 0 --use_dwconv --save_results


# Train Att_only
python main.py --model EDSR_VARIANTS --scale 2 --patch_size 64  --batch_size 32 --save sr_attention_x2 --epochs 350 --n_threads 0 --use_attention --save_results
# Test Baseline
#python main.py --model EDSR --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --n_threads 0 --self_ensemble

#Test Att_only
#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/sr_attention_x2/model/model_best.pt --test_only --self_ensemble --n_threads 0 --use_attention --n_resblocks 8

#Test Conv
#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/lumisr_dwconv_x2/model/model_best.pt --test_only --self_ensemble --n_threads 0 --use_dwconv

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_x2_from_base --pre_train ../models/edsr_baseline_x2-1bc95232.pt --epochs 300 --n_threads 0 --use_dwconv --save_results

