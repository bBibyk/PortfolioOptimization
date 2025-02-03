import os
import justetf_scraping as js
import numpy as np
import pandas as pd

script_folder = os.path.dirname(os.path.abspath(__file__))

df = js.load_overview(strategy="epg-longOnly", enrich=True)
# df = df.query('(replication=="Full replication" or replication=="Physically backed")')
df = df.query('age_in_years>=3')
df = df.dropna(subset=["last_three_years_volatility"])
df.to_csv(script_folder+"/data.csv")