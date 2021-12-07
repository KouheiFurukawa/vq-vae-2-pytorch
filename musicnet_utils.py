import pandas as pd

csv_input = pd.read_csv(filepath_or_buffer='/data/uni0/users/furukawa/musicnet_metadata.csv', sep=",")
genre_to_id = {
    'Solo Piano': 0, 'String Quartet': 1, 'Accompanied Violin': 2, 'Piano Quartet': 3, 'Accompanied Cello': 4,
    'String Sextet': 5, 'Piano Trio': 6, 'Piano Quintet': 7, 'Wind Quintet': 8, 'Horn Piano Trio': 9, 'Wind Octet': 10,
    'Clarinet-Cello-Piano Trio': 11, 'Pairs Clarinet-Horn-Bassoon': 12, 'Clarinet Quintet': 13, 'Solo Cello': 14,
    'Accompanied Clarinet': 15, 'Solo Violin': 16, 'Violin and Harpsichord': 17, 'Viola Quintet': 18, 'Solo Flute': 19,
    'Wind and Strings Octet': 20
}
song_name_to_genre = {}
for idx, row in csv_input.iterrows():
    genre = row['ensemble']
    song_id = str(row['id'])
    song_name_to_genre[song_id] = genre

def musicnet_label(file_path):
    filename = file_path.split('/')[-1][:4]
    return genre_to_id[song_name_to_genre[filename]]
