import pandas as pd
import pathlib

curr_dir = pathlib.Path(__file__).parent

# Read the CSV file
df = pd.read_csv(curr_dir / "results" / "results.csv")

# Replace <br> with newlines in the 'riddle' column
df['riddle'] = df['riddle'].str.replace('<br>', '\n')
df['riddle'] = df['riddle'].str.replace('<br/>', '\n')
df['riddle'] = df['riddle'].str.replace('<br />', '\n')

# Save the cleaned data back to CSV
df.to_csv(curr_dir / "results" / "results.csv", index=False)
