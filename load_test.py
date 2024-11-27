import pandas as pd

# Load the test data
test_fnc = pd.read_csv('./data/Test/test_FNC.csv')
test_sbm = pd.read_csv('./data/Test/test_SBM.csv')

# Merge the test features
test_data = pd.merge(test_fnc, test_sbm, on='Id')

# Load it into a new csv file
test_data[:30].to_csv('testing-files/test1.csv', index = False)
test_data[1200:1220].to_csv('testing-files/test2.csv', index = False)
test_data[100:115].to_csv('testing-files/test3.csv', index = False)
test_data[167:285].to_csv('testing-files/test4.csv', index = False)