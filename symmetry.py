import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []

    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []

        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)

        path_XYs.append(XYs)

    return path_XYs

def reflect_points(points, axis, center):
    reflected_points = np.copy(points)
    if axis == 'vertical':
        reflected_points[:, 0] = 2 * center[0] - points[:, 0]
    elif axis == 'horizontal':
        reflected_points[:, 1] = 2 * center[1] - points[:, 1]
    else:
        raise ValueError("Axis must be 'vertical' or 'horizontal'")
    return reflected_points

def is_symmetric(points, axis):
    center = np.mean(points, axis=0)
    reflected_points = reflect_points(points, axis, center)
    return np.allclose(points, reflected_points, atol=1e-2)

def plot_points_with_symmetry(points, axis, is_symmetric):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c='b', label='Original Points')
    center = np.mean(points, axis=0)
    if axis == 'vertical':
        plt.axvline(x=center[0], color='g' if is_symmetric else 'r', linestyle='--', label='Symmetry Line')
    elif axis == 'horizontal':
        plt.axhline(y=center[1], color='g' if is_symmetric else 'r', linestyle='--', label='Symmetry Line')
    plt.title('Symmetry Detection')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def save_results_to_csv(points, is_symmetric, output_path):
    df = pd.DataFrame(points, columns=['x', 'y'])
    df['is_symmetric'] = is_symmetric
    df.to_csv(output_path, index=False)

def main(csv_path, axis, output_csv_path):
    path_XYs = read_csv(csv_path)
    # Flatten the list of arrays and create a DataFrame
    data_frames = []
    for path in path_XYs:
        for segment in path:
            df = pd.DataFrame(segment, columns=['x', 'y'])
            data_frames.append(df)
    final_df = pd.concat(data_frames, ignore_index=True)
    # Extract x and y coordinates
    x = final_df['x'].values
    y = final_df['y'].values
    all_points = np.column_stack((x, y))
    symmetric = is_symmetric(all_points, axis)
    plot_points_with_symmetry(all_points, axis, symmetric)
    save_results_to_csv(all_points, symmetric, output_csv_path)
    print(f'Symmetry along {axis}: {symmetric}')

csv_path = str(input("Enter the path to csv "))
axis = str(input("Enter the line of symmetry horizontal/vertical "))  
output_csv_path = 'symmetry_results.csv'
main(csv_path, axis, output_csv_path)

