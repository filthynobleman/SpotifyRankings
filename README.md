# Spotify Rankings

A machine learning project based on the *Kaggle* dataset for the <a href="https://www.kaggle.com/edumucelli/spotifys-worldwide-daily-song-ranking">Top 200 Songs on Spotify</a>.  
The aim of the project is to perform some kinds of predictions using this dataset.  
The following kinds of prediction have been implemented:
* *Top Songs* prediction: this kind of prediction can be used to predict, knowing a song, its artist and a date, if that song will be in the top list of the Spotify's song in that date, limiting the search to a certain country or region. The length of the top list is variable and can be supplied to the predictor.
* *Today's Streams* prediction: this kind of prediction can be used to predict, knowing a song, its artist, a date and the streams of the previous day in which the song has obtained a position in the Spotify's top 200 in the same country, how many streams it will achieve in the given date.