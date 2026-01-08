import pandas as pd
import numpy as np

# Load the pickle file
df = pd.read_pickle(r'C:\Users\saisa\OneDrive\Desktop\main\main\pickle\y_val.pkl')

# Print the contents and shape
print(df)
print(df.shape)

# Save to .txt
with open(r'C:\Users\saisa\OneDrive\Desktop\main\main\pickle\dataset\y_val.txt', 'w', encoding='utf-8') as f:
    # If it's a NumPy array
    if isinstance(df, np.ndarray):
        f.write(np.array2string(df))
    else:
        f.write(str(df))
