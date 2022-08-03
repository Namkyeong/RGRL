cd ../
python main.py --device 0 --embedder RGRL --layers 512 256 --pred_hid 512 --lr 0.001 --epochs 1500 --sample 2048 --topk 2 --dataset wikics --aug_params 0.2 0.1 0.2 0.3 --temp_t 0.05 --temp_t_diff 1.0 --alpha 0.99 --beta 0.01 --lam 1.0 --eval_freq 5
python main.py --device 0 --embedder RGRL --layers 256 128 --pred_hid 512 --lr 0.001 --epochs 15000 --sample 512  --topk 4 --dataset computers --aug_params 0.2 0.1 0.5 0.4 --temp_t 0.01 --temp_t_diff 1.0 --alpha 0.999 --beta 0.1  --lam 1.0 --eval_freq 5
python main.py --device 0 --embedder RGRL --layers 512 256 --pred_hid 512 --lr 0.001 --epochs 5000 --sample 512  --topk 4 --dataset photo --aug_params 0.1 0.2 0.4 0.1 --temp_t 0.01 --temp_t_diff 1.0 --alpha 0.9 --beta 0.1  --lam 1.0 --eval_freq 5
python main.py --device 0 --embedder RGRL --layers 512 256 --pred_hid 512 --lr 0.001 --epochs 300 --sample 1024 --topk 8 --dataset cs --aug_params 0.3 0.4 0.3 0.2 --temp_t 0.01 --temp_t_diff 0.01 --alpha 0.9 --beta 0.0  --lam 1.0 --eval_freq 5
python main.py --device 0 --embedder RGRL --layers 256 128 --pred_hid 512 --lr 0.005 --epochs 400 --sample 256  --topk 8 --dataset physics --aug_params 0.1 0.4 0.4 0.1 --temp_t 0.01 --temp_t_diff 0.1 --alpha 0.9 --beta 0.01 --lam 0.7 --eval_freq 5

python main.py --device 0 --embedder RGRL --layers 256 128 --pred_hid 256 --lr 0.001 --epochs 400 --sample 128 --topk 2 --dataset cora --aug_params 0.3 0.4 0.2 0.4 --temp_t 0.01 --temp_t_diff 1.0 --alpha 0.9 --beta 0.1 --lam 1.0 --eval_freq 5
python main.py --device 0 --embedder RGRL --layers 512 256 --pred_hid 512 --lr 0.005 --epochs 50 --sample 128 --topk 2 --dataset citeseer --aug_params 0.3 0.2 0.2 0.0 --temp_t 0.01 --temp_t_diff 0.1 --alpha 0.9 --beta 0.1 --lam 1.0 --eval_freq 5
python main.py --device 0 --embedder RGRL --layers 512 256 --pred_hid 512 --lr 0.01 --epochs 2500 --sample 1024 --topk 2 --dataset pubmed --aug_params 0.0 0.2 0.4 0.1 --temp_t 0.01 --temp_t_diff 0.1 --alpha 0.9 --beta 0.0 --lam 1.0 --eval_freq 5
python main.py --device 0 --embedder RGRL --layers 512 256 --pred_hid 512 --lr 0.0001 --epochs 2000 --sample 8192 --topk 2 --dataset corafull --aug_params 0.5 0.5 0.5 0.5 --temp_t 0.01 --temp_t_diff 0.1 --alpha 0.9 --beta 0.0 --lam 1.0 --eval_freq 5