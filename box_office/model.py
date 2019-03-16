
import pandas
import plot
import numpy
import clean
import ast
import expand

x_train = pandas.read_csv('train.csv')
x_train = expand.expand_columns(x_train)
x_test = pandas.read_csv('test.csv')
x_test = expand.expand_columns(x_test)

#expand.expand_columns(x_test)

#genres = []

#for r in x_train.iterrows():
#	try:
#		gs = r[1]['original_language'] #ast.literal_eval(r[1]['original_language'])
#		#print(gs)
#		for g in gs:
#			genres.append(gs) #g['name'])
#	except:
#		pass

#for r in x_test.iterrows():
#	try:
#		gs = r[1]['original_language'] #ast.literal_eval(r[1]['original_language'])
#		for g in gs:
#			genres.append(gs) #['name'])
#	except:
#		pass

#for z in set(genres): print(z)

features = [
	{'name': 'belongs_to_collection', 'drop': True},	
	{'name': 'budget', 'na_val': 0}, #TODO this is wrong
	{'name': 'genres', 'drop': True},
	{'name': 'homepage', 'drop': True},
	{'name': 'imdb_id', 'drop': True},
	{'name': 'original_langage', 'drop': True},
	{'name': 'original_title', 'na_val': 'None'},
	{'name': 'overview', 'drop': True},
	{'name': 'popularity'},
	{'name': 'poster_path', 'drop': True},
	{'name': 'production_companies', 'drop': True},
	{'name': 'production_countries', 'drop': True},
	{'name': 'release_date'},
	{'name': 'runtime'},
	{'name': 'spoken_languages', 'drop': True},
	{'name': 'status', 'drop': True},
	{'name': 'tagline', 'drop': True},
	{'name': 'title', 'drop': True},
	{'name': 'Keywords', 'drop': True},
	{'name': 'cast', 'drop': True},
	{'name': 'crew', 'drop': True},

	{'name': 'collection', 'na_val': 'None'},
	{'name': 'adventure', 'na_val': 'None'},
	{'name': 'animation', 'na_val': 'None'},
	{'name': 'crime', 'na_val': 'None'},
	{'name': 'horror', 'na_val': 'None'},
	{'name': 'comedy', 'na_val': 'None'},
	{'name': 'romance', 'na_val': 'None'},
	{'name': 'drame', 'na_val': 'None'},
	{'name': 'foreign', 'na_val': 'None'},
	{'name': 'war', 'na_val': 'None'},
	{'name': 'science_fiction', 'na_val': 'None'},
	{'name': 'family', 'na_val': 'None'},
	{'name': 'thriller', 'na_val': 'None'},
	{'name': 'action', 'na_val': 'None'},
	{'name': 'western', 'na_val': 'None'},
	{'name': 'music', 'na_val': 'None'},
	{'name': 'history', 'na_val': 'None'},
	{'name': 'tv_movie', 'na_val': 'None'},
	{'name': 'documentary', 'na_val': 'None'},
	{'name': 'fantasy', 'na_val': 'None'},
	{'name': 'mystery', 'na_val': 'None'},
	{'name': 'language', 'na_val': 'None'},
	{'name': 'director', 'na_val': 'None'},
]


cleaner = clean.Cleaner(x_train, x_test)
cleaner.clean(features)

print(cleaner.nan_count(cleaner.x_train))
print(cleaner.nan_count(cleaner.x_test))










