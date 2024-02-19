
import torch
import numpy as np
from main import test, count_parameters
from faceformer import Faceformer
from data_loader import get_dataloaders
import os
import pickle




def lip_max_l2(vertice_dim, predict, real, mask):
    """
    This is the lip sync metric used in the faceformer paper
    """
    predict = torch.as_tensor(predict)
    real = torch.as_tensor(real)
    mask = torch.as_tensor(mask)

    mask = mask.to(real.device)
    lip_pred = predict * mask
    lip_real = real * mask

    pred_verts_mm = lip_pred.view(-1, vertice_dim//3, 3) * 1000.0
    gt_verts_mm = lip_real.view(-1, vertice_dim//3, 3) * 1000.0

    diff_in_mm = pred_verts_mm - gt_verts_mm
    l2_dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
    max_l2_error_lip_vert, idx = torch.max(l2_dist_in_mm, dim=-1)
    mean_max_l2_error_lip_vert = torch.mean(max_l2_error_lip_vert)
    return mean_max_l2_error_lip_vert


def main():
    import argparse
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument('--model_path', type=str, default='vocaset/save_backup/100_model.pth')
    parser.add_argument('--test_random_initialisation', type=bool, action='store_true', default=False)
    args = parser.parse_args()

    #build model
    model = Faceformer(args)

    #load model
    if args.test_random_initialisation:
        print('WARNING: testing random initialisation! results will be measured on a RANDOM model')
    else:
        print(f'loading model: {args.model_path}')
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict, strict=True)
        print("model parameters: ", count_parameters(model))
    
    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    #load data
    dataset = get_dataloaders(args)

    #save the test outputs to result/ folder
    print('testing model')
    test(args, model, dataset["test"], epoch=args.max_epoch)

    #evaluate outputs
    identity_to_result = {}

    pred_path = os.path.join(args.dataset, args.result_path)
    for file in os.listdir(pred_path):  
        if file.endswith("npy"):
            print("evaluating: ", file)

            predicted_vertices_path = os.path.join(pred_path,file)            
            predicted_vertices = np.load(predicted_vertices_path)

            sentence_name, identity = file.split('_condition_')
            gs_vertices_path = os.path.join(args.dataset, args.vertices_path, f'{sentence_name}.npy')
            gs_vertices = np.load(gs_vertices_path)[::2,:] #takes every other frame, essentialy             
            min_len = min([predicted_vertices.shape[0], gs_vertices.shape[0]])
            predicted_vertices = predicted_vertices[:min_len]
            gs_vertices = gs_vertices[:min_len]

            from FLAMEModel.flame_masks import get_flame_mask
            mask = get_flame_mask()
            lips_idxs = mask.lips
            lip_mask = torch.zeros((1, 5023, 3))
            lip_mask[0, lips_idxs] = 1.0
            lip_mask = lip_mask.view(1, -1)
            result = lip_max_l2(args.vertice_dim, predicted_vertices, gs_vertices, lip_mask)
            if identity in identity_to_result:
                identity_to_result[identity].append(result)
            else:
                identity_to_result[identity] = [result]
    all_identities = []
    for identity in identity_to_result:
        all_identities.extend(identity_to_result[identity])
        average_max_error = np.mean(identity_to_result[identity])
        print(f'{identity}: {average_max_error}')
    print(f'all_identities: {np.mean(all_identities)}')

            


if __name__=="__main__":
    main()