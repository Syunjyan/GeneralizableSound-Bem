import numpy as np

def generate_circle_coordinates(num_points=500):

    theta = np.ones(num_points) * 0.4
    # theta = np.linspace(0, 1, num_points)
    # theta = np.concatenate((np.ones(num_points//2) * 0.25, np.ones(num_points//2) * 0.75), axis=0)
    rs = np.ones(num_points) * 0.8
    # phi = np.concatenate((np.linspace(0, 1, num_points//2), np.linspace(0, 1, num_points//2)), axis=0)
    phi = np.linspace(0, 1, num_points)
    coordinates = {'x' : np.vstack((rs, theta, phi)).T, 'fps' : 30}
    return coordinates

def save_coordinates_to_npz(file_path, coordinates):
    np.savez(file_path, **coordinates)

if __name__ == "__main__":
    coordinates = generate_circle_coordinates()
    save_coordinates_to_npz('dataset/fix/animation_data.npz', coordinates)
    print("Coordinates saved to animation_data.npz")