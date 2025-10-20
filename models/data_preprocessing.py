import pandas as pd

def data_processing():
    """main function"""
    data_raw = pd.read_csv('data/raw/faithful.csv')
    data_new = data_raw.iloc[:,1]
    data_new.to_csv('data/processed/faithful.csv', index=False)
    data = pd.read_csv('data/processed/faithful.csv').to_numpy().flatten()

if __name__ == '__main__':
    data_processing()
