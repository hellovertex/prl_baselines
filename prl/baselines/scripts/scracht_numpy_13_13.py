import numpy as np
import pandas as pd
probas = np.random.random((13,13))
index=cols=["A", "K", "Q", "J", "T", "9",
            "8", "7", "6", "5", "4", "3", "2"]
df = pd.DataFrame(probas, index=index, columns=cols)
df.to_csv('probas.csv')
print(df)