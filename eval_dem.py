from models.autoencoder import DemAutoencoder
from data.dems import DemData

import cppimport.import_hook
from grid_planner import grid_planner

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np

import argparse
import os


def get_predictions(name='test', ckpt_path='./model.ckpt'):
    dataset = DemData(split=name)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0, pin_memory=True)
    model = DemAutoencoder(resolution=(dataset.img_size, dataset.img_size))
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    model.eval()
    predictions_dem = []
    predictions_focal = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            dem, rgb, sg, focal = batch
            inputs = torch.cat([rgb, sg], dim=1)
            predictions = (model(inputs) + 1) / 2
            predictions_dem.append(predictions[:, 0].numpy())
            predictions_focal.append(predictions[:, 1].numpy())
    predictions_dem = np.stack(predictions_dem, axis=0)
    predictions_focal = np.stack(predictions_focal, axis=0)
    np.savez(name + '_predictions.npz', dem=predictions_dem, focal=predictions_focal)
    print('Saved predictions to ' + name + '_predictions.npz')
    return predictions_dem, predictions_focal


def get_metrics(name='test', ckpt_path='./model.ckpt'):
    source_data = np.load(name + '.npz')
    source_focal = np.load(name + '_focal.npz')
    gt_dem = source_data['dem']
    starts = source_focal['start']
    goals = source_focal['goal']
    gt_focal = source_focal['focal']
    if os.path.exists(name + '_predictions.npz'):
        print('loading predictions')
        predictions = np.load(name + '_predictions.npz')
        predictions_dem = predictions['dem']
        predictions_focal = predictions['focal']
    else:
        predictions_dem, predictions_focal = get_predictions(name, ckpt_path)
    gt_dem_num = []
    pred_dem_num = []
    pred_focal_num = []    
    for i in tqdm(range(len(gt_dem))):
        for j in range(10):
            # search with A* and gt-dem
            planner = grid_planner(gt_dem[i][0].tolist())
            gt_dem_path = planner.find_path(starts[i][j], goals[i][j])
            gt_dem_num.append(planner.get_num_expansions())
            # search with A* and pred-dem
            planner = grid_planner((predictions_dem[i][j] * 255.).tolist())
            pred_dem_path = planner.find_path(starts[i][j], goals[i][j])
            pred_dem_num.append(planner.get_num_expansions())
            # focal search with predicted dem and focal values
            planner = grid_planner((predictions_dem[i][j] * 255.))
            pred_focal_path = planner.find_focal_path_reexpand(starts[i][j], goals[i][j], predictions_focal[i][j].tolist())
            pred_focal_num.append(planner.get_num_expansions())

    gt_dem_num = np.array(gt_dem_num)
    pred_dem_num = np.array(pred_dem_num)
    pred_focal_num = np.array(pred_focal_num)

    focal2pred_ratio_mean = (pred_focal_num / pred_dem_num).mean()
    pred2gt_ratio_mean = (pred_dem_num / gt_dem_num).mean()
    general_ratio_mean = (pred_focal_num / gt_dem_num).mean()

    print(f'Focal2pred ratio: {focal2pred_ratio_mean:.3f}')
    print(f'Pred2gt ratio: {pred2gt_ratio_mean:.3f}')
    print(f'General ratio:{general_ratio_mean:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./weights/dem_64.ckpt')

    args = parser.parse_args()
    get_metrics(ckpt_path=args.ckpt_path, name='./test')
