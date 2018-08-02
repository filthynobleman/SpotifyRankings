'''
This module contains informations about the dataset used by the program.
Notice that even if this module is intentionally left as general as possible, it implicitly
infers the structure of the dataset.
Also, notice that this module has been written with in mind the "Spotify's top 200 songs"
dataset from Kaggle.
The source of the dataset is:
https://www.kaggle.com/edumucelli/spotifys-worldwide-daily-song-ranking
'''

DATE_COLUMN = "Date"
'''
This string represents the name of the column containing the date in the dataset.
It can be supplied by the user. Default value is "Date".
'''
TRACKNAME_COLUMN = "Track Name"
'''
This string represents the name of the column containing the track name in the dataset.
It can be supplied by the user. Default value is "Track Name".
'''
ARTIST_COLUMN = "Artist"
'''
This string represents the name of the column containing the artist name in the dataset.
It can be supplied by the user. Default value is "Artist".
'''
POSITION_COLUMN = "Position"
'''
This string represents the name of the column containing the occupied position in the dataset.
It can be supplied by the user. Default value is "Position".
'''
REGION_COLUMN = "Region"
'''
This string represents the name of the column containing the region code in the dataset.
It can be supplied by the user. Default value is "Region".
'''
URL_COLUMN = "URL"
'''
This string represents the name of the column containing the song's URL in the dataset.
It can be supplied by the user. Default value is "URL".
'''
STREAMS_COLUMN = "Streams"
'''
This string represents the name of the column containing the song's streams count in the
dataset. It can be supplied by the user. Default value is "Streams".
'''
PREVSTREAMS_COLUMN = "Previous Streams"
'''
This string represents the name of the column containing the song's streams count in the
last day when it has scored a position in the top 200 for the same country.
It can be supplied by the user. Default value is "Previous Streams".
'''
ISINTOP_COLUMN = "Is In Top"
'''
This string represents the name of the column containing the boolean value that describe if
the track of the artist in that day was in the top list of the tracks for the country.
It can be supplied by the user. Default value is "Is In Top".
'''