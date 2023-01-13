import pandas as pd
import numpy as np

df = pd.read_csv('./dsl_data/development.csv')

df['labels'] = df['action'].astype(str) + df['object'].astype(str)

distinct_values = df['labels'].unique()

# print(distinct_values)

result = 'LABELS = ['
for value in distinct_values:
    result += "'" + str(value) + "', "

result = result[:-2] + ']\n' # lazy workaround, the last label has a comma that is bad.. this is also bad.
# result += ']\n'

# Open the file in read mode
with open("preprocessing.py", "r") as file:
    # Read the contents of the file into a list
    lines = file.readlines()

# Update the 3rd line
lines[3] = result
lines[4] = "# This is the file genarated that has the Labels that i must use for training\n"

# Open the file in write mode
with open("preprocessing.py", "w") as file:
    # Write the updated lines back to the file
    file.writelines(lines)
