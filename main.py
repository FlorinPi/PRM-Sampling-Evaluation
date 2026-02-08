# Main and helper function

from PIL import Image
import numpy as np
from PRM import PRM
import os

import matplotlib.pyplot as plt


def load_map(file_path, resolution_scale):
    ''' Load map from an image and return a 2D binary numpy array
        where 0 represents obstacles and 1 represents free space
    '''
    # Load the image with grayscale
    img = Image.open(file_path).convert('L')
    # Rescale the image
    size_x, size_y = img.size
    new_x, new_y = int(size_x*resolution_scale), int(size_y*resolution_scale)
    # Pillow removed Image.ANTIALIAS in newer versions; use Resampling.LANCZOS when available
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = getattr(Image, 'LANCZOS', Image.BICUBIC)
    img = img.resize((new_x, new_y), resample_filter)

    map_array = np.asarray(img, dtype='uint8')

    # Get bianry image
    threshold = 127
    map_array = 1 * (map_array > threshold)

    # Result 2D numpy array
    return map_array


def calculate_path_statistics(PRM_planner, nr_of_tries):
    # print("Paths lengths over tries:", PRM_planner.paths_length)
    # Count failed runs (zeros) and compute average over successful runs only
    lengths = PRM_planner.paths_length
    fail_count = sum(1 for L in lengths if L == 0)
    success_count = len(lengths) - fail_count
    if success_count > 0:
        avg_path_length = sum(L for L in lengths if L != 0) / success_count
    else:
        avg_path_length = 0
    print(
        f"Runs: {len(lengths)}, Successes: {nr_of_tries - fail_count}, Failures: {fail_count}")
    print(f"Average path length (successful runs): {avg_path_length}")

    # print("Graph build times over tries:", PRM_planner.build_time)
    avg_build_time = sum(PRM_planner.build_time) / \
        len(PRM_planner.build_time)
    print(f"Average graph build time: {avg_build_time} seconds")

    return fail_count, avg_path_length, avg_build_time


def run_trials(PRM_planner, start, goal, n_pts, k, std_dev, p_random, sampling_method_str, nr_of_tries, show_drawing):

    # Search with PRM
    for i in range(nr_of_tries):
        PRM_planner.sample(n_pts, k, std_dev, p_random, sampling_method_str)
        PRM_planner.search(start, goal, k, show_drawing)

    # Calculate statistics
    fail_count, avg_path_length, avg_build_time = calculate_path_statistics(
        PRM_planner, nr_of_tries)

    if PRM_planner.best_length != np.inf:
        best_path_length = PRM_planner.best_length
        best_build_time = PRM_planner.best_time
    else:
        best_path_length = None
        best_build_time = None

    return {
        'runs': nr_of_tries,
        'fail_count': fail_count,
        'success_count': nr_of_tries - fail_count,
        'avg_path_length': avg_path_length,
        'avg_build_time': avg_build_time,
        'best_path_length': best_path_length,
        'best_build_time': best_build_time
    }


def save_image(PRM_planner, scenario_name, method_name, param_name, param_value):
    # Save result image
    img_path = os.path.join(
        'results', f'{scenario_name}', f'{method_name}', f'{param_name} {param_value}.png')

    # Draw best result
    if PRM_planner.best_length != np.inf:
        print("Best path length:", PRM_planner.best_length)
        print("Best build time:", PRM_planner.best_time)
        PRM_planner.draw_map(PRM_planner.best_samples,
                             PRM_planner.best_path, PRM_planner.best_graph, save_path=img_path)
    else:
        print("No valid path found in any of the tries.")
        PRM_planner.draw_map(PRM_planner.samples,
                             PRM_planner.path, PRM_planner.graph, save_path=img_path)


def save_summary(results, scenario_name, method_name, param_name, param_value):
    # Save summary to text file
    summary_lines = []
    summary_lines.append(
        f"Runs: {results['runs']}, Successes: {results['success_count']}, Failures: {results['fail_count']}")
    summary_lines.append(
        f"Average path length: {results['avg_path_length']}")
    summary_lines.append(
        f"Average build time: {results['avg_build_time']} seconds")
    summary_lines.append(
        f"Best path length: {results['best_path_length']}")
    summary_lines.append(
        f"Best buld time: {results['best_build_time']} seconds")
    summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(
        'results', f'{scenario_name}', f'{method_name}', f'{param_name} {param_value} output.txt')
    try:
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        print(f"Saved summary to {summary_path}")
    except Exception as e:
        print("Failed to write summary file:", e)


if __name__ == "__main__":
    # Hyperparameters
    n_pts_list = [100, 500, 1000]
    k_list = [5, 10, 15]
    std_dev_list = [10, 20, 40]
    p_random_list = [0.1, 0.3, 0.5]

    defaults = {
        'N': n_pts_list[1],
        'k': k_list[1],
        's': std_dev_list[1],
        'p': p_random_list[1],
    }

    param_lists = {
        'N': n_pts_list,
        'k': k_list,
        's': std_dev_list,
        'p': p_random_list,
    }

    # Flags
    show_drawing = False
    scenarios = [1, 2, 3, 4]
    # 1: random, 2: uniform, 3: gaussian, 4: bridge
    sampling_methods = [1, 2, 3, 4]

    scenario_dir_name = {
        1: "Scenario1",
        2: "Scenario2",
        3: "Scenario3",
        4: "Scenario4",
    }

    sampling_method_dir_name = {
        1: "Random",
        2: "Grid",
        3: "Gaussian",
        4: "Bridge",
    }

    # Load the map (640 x 480)
    start = (400, 60)
    goal = (20, 600)

    for scenario in scenarios:
        # select scenario
        match scenario:
            case 1: map_array = load_map("scenario1.jpg", 1.0)
            case 2: map_array = load_map("scenario2.jpg", 1.0)
            case 3: map_array = load_map("scenario3.jpg", 1.0)
            case 4: map_array = load_map("scenario4.jpg", 1.0)

        for sampling_method in sampling_methods:
            match sampling_method:
                case 1: sampling_method_str = "random"
                case 2: sampling_method_str = "uniform"
                case 3: sampling_method_str = "gaussian"
                case 4: sampling_method_str = "bridge"

            # Set number of tries per sampling method
            nr_of_tries = 100

            # Determine which parameters to sweep
            sweep_params = ['N', 'k', 's', 'p']
            if sampling_method in (1, 2):
                sweep_params = ['N', 'k']
                if sampling_method == 2:
                    nr_of_tries = 1  # grid is deterministic

            for param in sweep_params:
                for value in param_lists[param]:
                    current_params = defaults.copy()
                    current_params[param] = value

                    n_pts = current_params['N']
                    k = current_params['k']
                    std_dev = current_params['s']
                    p_random = current_params['p']

                    # Planning class
                    PRM_planner = PRM(map_array)

                    # Run trials
                    results = run_trials(PRM_planner, start, goal, n_pts, k, std_dev,
                                         p_random, sampling_method_str, nr_of_tries, show_drawing)

                    # Create
                    os.makedirs(os.path.join(
                        'results', f'{scenario_dir_name[scenario]}', f'{sampling_method_dir_name[sampling_method]}'), exist_ok=True)

                    # Save image
                    save_image(
                        PRM_planner, scenario_dir_name[scenario], sampling_method_dir_name[sampling_method], param, current_params[param])

                    # Save summary
                    save_summary(
                        results, scenario_dir_name[scenario], sampling_method_dir_name[sampling_method], param, current_params[param])
