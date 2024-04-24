# for speaker in
# for train_subject in FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA; do
# test_subject=FaceTalk_170809_00138_TA #these are the test speakers!
test_subject=FaceTalk_170731_00024_TA

base_model_path=/home/paz/code/phd/FaceFormer/vocaset/save_federated/clients_1_max_epoch_10_rounds_25_aggr_avg_data_split_vertical_train_idx_-1_model_imitator/model_round_24.pth
# base_model_path=/home/paz/code/phd/FaceFormer/vocaset/save_federated/clients_1_max_epoch_5_aggr_avg_data_split_vertical_train_idx_-1_model_imitator_gen/model_round_4.pth 
# base_model_path=/home/paz/code/phd/FaceFormer/vocaset/save_federated/clients_8_max_epoch_10_aggr_avg_data_split_vertical_train_idx_-1_model_imitator_gen/model_round_25.pth

# max_epoch=2
max_epoch=100
max_rounds=1

# aggr=avg
aggr=mask #just to initialise with all speakers

for test_subject in FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA; do

    initialise_idx=2 #same as paper, set as train_idx for initialisation
    echo "doing test subject ${test_subject}";
    #train general (not necessary for each speaker)
    # python flwr_client.py --max_epoch 5 --max_rounds 5 --num_clients 1  --dataset vocaset --model imitator_gen --train_idx ${spkid};
    # gen > type1
    python flwr_client.py --max_epoch ${max_epoch} --max_rounds 1 --num_clients 1  --dataset vocaset --model imitator_stg01 --train_idx 0 --condition_idx ${initialise_idx} --train_subjects ${test_subject}  --data_split stg --aggr ${aggr} --base_model_path ${base_model_path};
    #type1 > type2
    python flwr_client.py --max_epoch ${max_epoch} --max_rounds 1 --num_clients 1  --dataset vocaset --train_subjects ${test_subject} --model imitator_stg02 --train_idx 0 --condition_idx ${initialise_idx} --data_split stg --aggr ${aggr} --base_model_path /home/paz/code/phd/FaceFormer/vocaset/save_federated/clients_1_max_epoch_${max_epoch}_aggr_${aggr}_data_split_stg_train_idx_0_model_imitator_stg01/model_round_0.pth;
    #eval type2
    python jp_test_model_voca.py --dataset vocaset  --model imitator --model_path /home/paz/code/phd/FaceFormer/vocaset/save_federated/clients_1_max_epoch_${max_epoch}_aggr_${aggr}_data_split_stg_train_idx_0_model_imitator_stg02/model_round_0.pth --train_subjects ${test_subject} --test_subject ${test_subject} --data_split stg;

    rm -r /home/paz/code/phd/FaceFormer/vocaset/save_federated/${test_subject}
    mkdir /home/paz/code/phd/FaceFormer/vocaset/save_federated/${test_subject}

    mv /home/paz/code/phd/FaceFormer/vocaset/save_federated/clients_1_max_epoch_${max_epoch}_aggr_${aggr}_data_split_stg_train_idx_0_model_imitator_stg02/ /home/paz/code/phd/FaceFormer/vocaset/save_federated/${test_subject}
done


