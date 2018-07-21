# import preprocessing as pp
# import pandas as pd
# from time import time

# start = time()
# data = pd.read_csv("data.csv")
# data.drop(columns = ['URL'], inplace = True)
# data = data[~data['Artist'].isna()]
# data = data[~data['Track Name'].isna()]
# data = data[data['Region'] == 'it']
# pp.encode_dates(data, inplace = True)
# print data.head()
# pp.encode_artists(data, inplace = True)
# print data.head()
# pp.encode_tracks(data, inplace = True)
# print data.head()
# print "Elapsed time: {}".format(time() - start)

import topsongs

topsongs.test_topsongs()