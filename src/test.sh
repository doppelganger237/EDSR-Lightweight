#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_attention_x2_full --reset --epochs 300 --n_threads 0 --use_dwconv --use_attention --save_results 

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_x2 --load lumisr_dwconv_x2 --pre_train ../experiment/lumisr_dwconv_x2/model/model_latest.pt --epochs 300 --n_threads 0 --use_dwconv --save_results 

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_attention_x2 --epochs 300 --n_threads 0 --use_attention --save_results 
#python main.py --model EDSR_VARIANTS --scale 2 --pre_train ../experiment/lumisr_dwconv_attention_x2_full/model/model_best.pt --test_only --use_dwconv --use_attention --n_colors 1



# Train Att_only
python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96  --n_resblocks 8 --save sr_attention_x2 --reset --epochs 300 --n_threads 4 --use_attention --save_results

# Test Baseline
#python main.py --model EDSR --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --save edsr_x2_only_y_bench --save_results --n_threads 0 --only_y

#Test Att_only
#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/sr_attention_x2/model/model_best.pt --test_only --self_ensemble --save sr_attention_x2_only_y_benchmark --save_results --n_threads 0 --use_attention --only_y


#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_x2_from_base --pre_train ../models/edsr_baseline_x2-1bc95232.pt --epochs 300 --n_threads 0 --use_dwconv --save_results

