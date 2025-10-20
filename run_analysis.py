from models.data_preprocessing import data_processing
from models.mulkde_comparison import comparison
from models.mulkde_simu import simulation


data_processing()
print('Data proprocessed successfully!\n')

comparison()
print("Methods compared successfully!\n")

simulation()
print("Methods tested on Old Faithful dataset successfully!")
