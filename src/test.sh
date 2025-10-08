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
python main.py --data_test Set5+Set14 --scale 2 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --self_ensemble --save edsr_x2_bench --save_results --n_threads 0


