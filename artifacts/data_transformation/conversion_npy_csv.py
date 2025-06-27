import numpy as np
import pandas as pd

# Load the .npy file
npy_file_path = 'C:\\Users\\hp\\Desktop\\credit card\\credit_default_prediction_complete\\credit_default_prediction\\artifacts\\data_transformation\\test.npy'
data = np.load(npy_file_path)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as .csv
csv_file_path = npy_file_path.replace('.npy', '.csv')
df.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")