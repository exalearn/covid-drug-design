"""Make a table with the best-performing molecules for Table 1"""

import pandas as pd
import os

# Hard-code the order of rows/columns in the table
rewards = ['logP', 'QED', 'ic50', 'MO']
cols = ['ic50', 'QED', 'logP', 'SA']

# Load in the molecules and get the best ones
for reward in rewards:
    # Load in the molecules
    all_data = pd.read_csv(os.path.join('reference-runs', reward, 'molecules.csv'))

    # Get the top 3 by their "score"
    top_data = all_data.sort_values('reward', ascending=False).head(3)
    print(f'{reward} top mol: {top_data["smiles"].iloc[0]}')

    # Make the row
    row = f'{reward}'
    for col in cols:
        row += " & " + " & ".join(f"{x: .2f}" for x in top_data[col])
    print(row)
