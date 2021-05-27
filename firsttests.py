import pandas

# read movie lens
movies = pandas.read_csv('movies.csv')
ratings = pandas.read_csv('ratings.csv')
tags = pandas.read_csv('tags.csv')
rating_table = pandas.read_csv('rating_table.csv')

# split the genres per movie
m_info_map = []
genres = []
for index, movie in movies.iterrows():
    m_info_map.append([movie['genres'].split('|'), []])
    for g in movie['genres'].split('|'):
        if g not in genres:
            genres.append(g)

# give the genres ids
genres.sort()
gen_dict = []
count = 0
for g in genres:
    gen_dict.append((g, count))
    count += 1
gen_dict = dict(gen_dict)

# create a datatable with compressed info
movie_map = movies['movieId'].values.tolist()
for index, row in tags.iterrows():
    i = movie_map.index(row['movieId'])
    r = row['tag']
    m_info_map[i][1].append(r)

for ind in range(len(m_info_map)):
    for ind2 in range(len(m_info_map[ind][0])):
        m_info_map[ind][0][ind2] = gen_dict[m_info_map[ind][0][ind2]]
    m_info_map[ind].insert(1, len(m_info_map[ind][1]))

# rating table
if False:
    user_map = [ratings['userId'][0]]
    rating_table = [[0.0 for i in range(len(movie_map))]]
    for index, rating in ratings.iterrows():
        if user_map[-1] == rating['userId']:
            rating_table[-1][movie_map.index(rating['movieId'])] = rating['rating']
        else:
            rating_table.append([0 for i in range(len(movie_map))])
            rating_table[-1][movie_map.index(rating['movieId'])] = rating['rating']
            user_map.append(rating['userId'])
    df2 = pandas.DataFrame(rating_table)
    df2.to_csv('rating_table.csv')

df = pandas.DataFrame(m_info_map)


