import numpy as np
import sys
import matplotlib.pyplot as plt

DATA_DIR = sys.argv[1]
HBIE_bempp = np.loadtxt(f"{DATA_DIR}/HBIE_bempp.txt")
CBIE_bempp = np.loadtxt(f"{DATA_DIR}/CBIE_bempp.txt")
HBIE_cuda = np.loadtxt(f"{DATA_DIR}/HBIE_cuda.txt")
CBIE_cuda = np.loadtxt(f"{DATA_DIR}/CBIE_cuda.txt")
HBIE_cuda_approx = np.loadtxt(f"{DATA_DIR}/HBIE_cuda_approx.txt")


plt.figure()
label_list = ["HBIE Bempp", "CBIE Bempp", "HBIE CUDA", "CBIE CUDA", "HBIE CUDA Approx"]
data_list = [HBIE_bempp, CBIE_bempp, HBIE_cuda, CBIE_cuda, HBIE_cuda_approx]
for idx in range(len(data_list)):
    wave_number = data_list[idx][:, 0]
    freqs = wave_number / (2 * np.pi) * 343.3
    plt.plot(freqs, data_list[idx][:, 1], label=label_list[idx])

plt.xlabel(f"Frequency")
plt.ylabel(f"Relative error")
plt.yscale(f"log")
plt.legend()
plt.xticks()
plt.savefig(f"{DATA_DIR}/unit_test_error.png", dpi=300)

plt.close()
plt.figure()

for idx in range(len(data_list)):
    wave_number = data_list[idx][:, 0]
    freqs = wave_number / (2 * np.pi) * 343.3
    plt.plot(freqs, data_list[idx][:, 2], label=label_list[idx])

plt.xlabel(f"Frequency")
plt.ylabel(f"Time cost (s)")
plt.yscale(f"log")
plt.legend()
plt.xticks()
plt.savefig(f"{DATA_DIR}/unit_test_time.png", dpi=300)
plt.close()
