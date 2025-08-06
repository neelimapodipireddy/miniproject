import h5py

try:
    with h5py.File('skin_tone_model.h5', 'r') as f:
        print("File is a valid HDF5 file.")
except Exception as e:
    print(f"Invalid HDF5 file: {e}")
