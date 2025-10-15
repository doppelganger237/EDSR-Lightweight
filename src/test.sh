python main.py --model EDSR_VARIANTS --scale 2 --patch_size 96 --save lumisr_dwconv_attention_x2_full --reset --epochs 200 --data_range 1-800/801-810 --n_threads 4 --use_dwconv --use_attention --save_results --n_colors 1
#python main.py --model EDSR_VARIANTS --scale 2 --pre_train ../experiment/lumisr_dwconv_attention_x2_full/model/model_best.pt --test_only --use_dwconv --use_attention --n_colors 1

# 把路径改成你本地的 baseline checkpoint 路径
#python main.py --model EDSR --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --self_ensemble --save edsr_x2_y_bench --save_results --n_threads 0 --only_y

# 把路径改成你本地的 baseline checkpoint 路径
#python main.py --model EDSR_VARIANTS --data_test Set5+Set14+B100+Urban100 --scale 2 --pre_train ../experiment/lumisr_dwconv_attention_x2_full/model/model_best.pt --test_only --self_ensemble --save lumisr_dwconv_attention_x2_y_benchmark --save_results --n_threads 0 --use_dwconv --use_attention --only_y
