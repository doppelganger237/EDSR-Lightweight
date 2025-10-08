#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --scale 2 --pre_train download --test_only --self_ensemble

# Test your own images
python main.py --data_test Demo --scale 4 --pre_train ../models/edsr_baseline_x2-1bc95232.pt --test_only --save_results