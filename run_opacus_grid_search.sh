#first activate opacus conda env

for delta in 0.001; do #1/dataset size = 0.002 
 for noise_multiplier in 0.1 1 10; do 
  for max_grad_norm in 0.1 0.01 0.001; do 
   python flwr_client.py --model imitator_gen --max_epoch 5 --max_rounds 10 --num_clients 1 --data_split vertical --aggr avg --dp opacus --delta ${delta} --noise_multiplier ${noise_multiplier} --max_grad_norm ${max_grad_norm} --save_path opacus_grid_search
  done
 done
done

