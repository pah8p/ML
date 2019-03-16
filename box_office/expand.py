
import pandas
import ast



def collection(r):
	try:
		_obj = ast.literal_eval(r['belongs_to_collection'])
	except ValueError:
		return 'None'

	collection = _obj[0]['name']
	return collection	


def genres(r, genre):
	try:
		_obj = ast.literal_eval(r['genres'])
	except ValueError:
		return 'None'

	for g in _obj:
		if g['name'] == genre:
			return 1
	else:
		return 0

def language(r):

	_map = {
		'it': 'Italian',
		#'nl': 'Netherlands',
		'fa': 'Farsi',
		#'cs': 'Czech Republic',
		#'te': 'Telugu',
		'pt': 'Portuguese',
		#'is': 'Iceland',
		#'ca': 'Canada',
		#'nb': 'Norsk',
		'vi': '',
		'th': '',
		'en': 'English',
		'de': 'German',
		'es': 'Spanish',
		'fr': 'French',
		'ar': 'Arabic',
		'cn': 'Chinese',
		'ja': 'Japanese',
		'ru': 'Russian',
	}
	
	_obj = r['original_language']
	
	try:
		return _map[_obj]
	except KeyError:
		return 'Other'
	
def production_companies(r):
	#TODO: Split Hollywood/Not Hollywood
	pass

def crew(r, job):
	try:
		_obj = ast.literal_eval(r['crew'])
	except ValueError:
		return 'None'

	for c in _obj:
		if c['job'] == job:
			return c['name']
	else:
		return 'None'

def actors(r, n):
	try:
		_obj = ast.literal_eval(r['crew'])
	except ValueError:
		return 'None'

	try:
		return _obj[n]['name']
	except IndexError:
		return 'None'

def expand_columns(data):
	data['collection'] = data.apply(lambda r: collection(r), axis=1)

	data['adventure'] = data.apply(lambda r: genres(r, 'Adventure'), axis=1)
	data['animation'] = data.apply(lambda r: genres(r, 'Animation'), axis=1)
	data['crime'] = data.apply(lambda r: genres(r, 'Crime'), axis=1)
	data['horror'] = data.apply(lambda r: genres(r, 'Horror'), axis=1)
	data['comedy'] = data.apply(lambda r: genres(r, 'Comedy'), axis=1)
	data['romance'] = data.apply(lambda r: genres(r, 'Romance'), axis=1)
	data['drame'] = data.apply(lambda r: genres(r, 'Drama'), axis=1)
	data['foreign'] = data.apply(lambda r: genres(r, 'Foreign'), axis=1)
	data['war'] = data.apply(lambda r: genres(r, 'War'), axis=1)
	data['science_fiction'] = data.apply(lambda r: genres(r, 'Science Fiction'), axis=1)
	data['family'] = data.apply(lambda r: genres(r, 'Family'), axis=1)
	data['thriller'] = data.apply(lambda r: genres(r, 'Thriller'), axis=1)
	data['action'] = data.apply(lambda r: genres(r, 'Action'), axis=1)
	data['western'] = data.apply(lambda r: genres(r, 'Western'), axis=1)
	data['music'] = data.apply(lambda r: genres(r, 'Music'), axis=1)
	data['history'] = data.apply(lambda r: genres(r, 'History'), axis=1)
	data['tv_movie'] = data.apply(lambda r: genres(r, 'TV Movie'), axis=1)
	data['documentary'] = data.apply(lambda r: genres(r, 'Documentary'), axis=1)
	data['fantasy'] = data.apply(lambda r: genres(r, 'Fantasy'), axis=1)
	data['mystery'] = data.apply(lambda r: genres(r, 'Mystery'), axis=1)

	data['language'] = data.apply(lambda r: language(r), axis=1)

	data['director'] = data.apply(lambda r: crew(r, 'Director'), axis=1)

	data['actor_1'] = data.apply(lambda r: actors(r, 0), axis=1)
	data['actor_2'] = data.apply(lambda r: actors(r, 1), axis=1)
	data['actor_3'] = data.apply(lambda r: actors(r, 2), axis=1)

	return data

