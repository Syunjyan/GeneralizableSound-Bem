import numpy as np

def generate_circle_coordinates(num_points=400, radius=0.4):

    theta = np.concatenate((np.zeros(100), np.linspace(0, 2 * np.pi, num_points)), axis=0)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(100 + num_points)
    coordinates = {'x' : np.vstack((x, y, z)).T, 'fps' : 30}
    return coordinates

def save_coordinates_to_npz(file_path, coordinates):
    np.savez(file_path, **coordinates)

if __name__ == "__main__":
    coordinates = generate_circle_coordinates()
    save_coordinates_to_npz('dataset/NeuPAT_new/fix/animation_data.npz', coordinates)
    print("Coordinates saved to animation_data.npz")