python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_attention_x2_full --reset --epochs 300 --data_range 1-800/801-810 --n_threads 0 --use_dwconv --use_attention --save_results 

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_x2 --load lumisr_dwconv_x2 --pre_train ../experiment/lumisr_dwconv_x2/model/model_latest.pt --epochs 300 --n_threads 0 --use_dwconv --save_results 

#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_attention_x2 --epochs 300 --n_threads 0 --use_attention --save_results 
#python main.py --model EDSR_VARIANTS --scale 2 --pre_train ../experiment/lumisr_dwconv_attention_x2_full/model/model_best.pt --test_only --use_dwconv --use_attention --n_colors 1

# 把路径改成你本地的 baseline checkpoint 路径
#python main.py --model EDSR --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --self_ensemble --save edsr_x2_bench_2 --save_results --n_threads 0 

# 把路径改成你本地的 baseline checkpoint 路径
#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/lumisr_dwconv_attention_x2_full/model/model_best.pt --test_only --self_ensemble --save lumisr_dwconv_attention_x2_benchmark --save_results --n_threads 0 --use_dwconv --use_attention

#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/lumisr_dwconv_x2/model/model_best.pt --test_only --self_ensemble --save lumisr_dwconv_x2_benchmark --save_results --n_threads 0 --use_dwconv


#python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_x2_from_base --pre_train ../models/edsr_baseline_x2-1bc95232.pt --epochs 300 --n_threads 0 --use_dwconv --save_results
