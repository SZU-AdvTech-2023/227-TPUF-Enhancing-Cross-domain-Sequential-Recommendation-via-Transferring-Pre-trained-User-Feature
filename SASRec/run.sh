nohup python -u main.py --dataset Books --dataset_path /home/temp_user/wudongqi/Archive/TPUF/data/v1/branch/v1.1/data/cross_data/ --maxlen 100 --model SASRec --l2_emb 0.0 --hidden_units 50 --num_epochs 500 --gpu 2 > logs/Source_Books_Target_Movies_and_TV.logs 2>&1 &

nohup python -u main.py --dataset Books --dataset_path /home/temp_user/wudongqi/Archive/TPUF/data/v1/branch/v1.2/data/cross_data/ --maxlen 100 --model SASRec --l2_emb 0.0 --hidden_units 50 --num_epochs 500 --gpu 2 > logs/Source_Books_Target_CDs_and_Vinyl.logs 2>&1 &

nohup python -u main.py --dataset Movies_and_TV --dataset_path /home/temp_user/wudongqi/Archive/TPUF/data/v1/branch/v1.3/data/cross_data/ --maxlen 100 --model SASRec --l2_emb 0.0 --hidden_units 50 --num_epochs 500 --gpu 2 > logs/Source_Movies_and_TV_Target_Books.logs 2>&1 &

nohup python -u main.py --dataset Movies_and_TV --dataset_path /home/temp_user/wudongqi/Archive/TPUF/data/v1/branch/v1.1/data/cross_data/ --maxlen 100 --model SASRec --l2_emb 0.001 --hidden_units 50 --num_epochs 500 --gpu 3 > logs/Target_Movies_and_TV.logs 2>&1 &

nohup python -u main.py --dataset CDs_and_Vinyl --dataset_path /home/temp_user/wudongqi/Archive/TPUF/data/v1/branch/v1.2/data/cross_data/ --maxlen 100 --model SASRec --l2_emb 0.001 --hidden_units 50 --num_epochs 500 --gpu 3 > logs/Target_CDs_and_Vinyl.logs 2>&1 &

nohup python -u main.py --dataset Books --dataset_path /home/temp_user/wudongqi/Archive/TPUF/data/v1/branch/v1.3/data/cross_data/ --maxlen 100 --model SASRec --l2_emb 0.001 --hidden_units 50 --num_epochs 500 --gpu 3 > logs/Target_Books.logs 2>&1 &