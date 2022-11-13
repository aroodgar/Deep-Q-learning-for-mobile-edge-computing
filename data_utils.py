import numpy as np
import pandas as pd

def save_numpy_to_csv(nparr, filename):
  # convert array into dataframe
  DF = pd.DataFrame(nparr)
  # save the dataframe as a csv file
  DF.to_csv(f"{filename}.csv")

def convert_csv_to_excel(ifile, ofile):
  # Reading the csv file
  df_new = pd.read_csv(f'{ifile}.csv')
  # saving xlsx file
  GFG = pd.ExcelWriter(f'{ofile}.xlsx')
  df_new.to_excel(GFG, index=False)
  GFG.save()

def test():
  x = np.arange(0.0,3.0,0.5)
  save_numpy_to_csv(x, 'otest')
  convert_csv_to_excel('otest', 'otest')

if __name__ == "__main__":
  test()