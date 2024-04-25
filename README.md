# Mono Video Reconstruction

CUDA_VISIBLE_DEVICES=1 python train_end_to_end.py --data_path /raid/compass/mobarak/datasets/Endonasal_Phantom/ --dataset endonasal --log_dir logs_wpretrained_all --load_weights_folder ../weights/Model_MIA --num_epochs 50

CUDA_VISIBLE_DEVICES=0 python evaluate_3d_reconstruction.py --data_path /raid/compass/mobarak/datasets/Endonasal_Phantom/ --eval_mono --load_weights_folder logs_wpretrained_all/mdp/models/weights_18 --visualize_depth --save_recon --dataset endonasal

CUDA_VISIBLE_DEVICES=0 python evaluate_3d_reconstruction.py --data_path /raid/compass/mobarak/datasets/Endonasal_Phantom/ --eval_mono --load_weights_folder ../weights/Model_MIA --visualize_depth --save_recon --dataset endonasal
