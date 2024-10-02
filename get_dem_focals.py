import argparse
import numpy as np
from tqdm import tqdm
import cppimport.import_hook
from grid_planner import grid_planner

def generate_tasks(num_tasks, grid_size):
    tasks = []
    while len(tasks) < num_tasks:
        coords = np.random.randint(0, grid_size, 4)
        if abs(coords[0] - coords[2]) + abs(coords[1] - coords[3]) > grid_size:
            tasks.append({'start': (coords[0], coords[1]), 'goal': (coords[2], coords[3])})
    return tasks

def get_focal_values(dem):
    results = []
    starts = []
    goals = []
    dem = dem[0]
    planner = grid_planner(dem.tolist())
    tasks = generate_tasks(10, dem.shape[0])
    for task in tasks:
        starts.append(np.array(task['start']))
        goals.append(np.array(task['goal']))
        results.append(planner.find_heatmap(task['start'], task['goal']))
    return np.stack(results)[:, None, :, :], np.stack(starts), np.stack(goals)

def proc_file(filename):
    split = filename[:-4]
    new_filename = split + '_focal.npz'
    focals, starts, goals = [], [], []
    dems = np.load(filename)['dem']
    for dem in tqdm(dems):
        focal, start, goal = get_focal_values(dem)
        focals.append(focal)
        starts.append(start)
        goals.append(goal)
    np.savez(new_filename, focal=np.stack(focals), start=np.stack(starts), goal=np.stack(goals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenames', nargs='+', type=str, default=['./val.npz', './train.npz', './test.npz'])
    args = parser.parse_args()
    for filename in args.filenames:
        proc_file(filename)
        
    
if __name__ == '__main__':
    main()
