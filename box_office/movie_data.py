
import ast
import datetime
import pandas
import math

language_map = {
	'it': 'Italian',
	#'nl': 'Netherlands',
	'fa': 'Farsi',
	#'cs': 'Czech Republic',
	#'te': 'Telugu',
	'pt': 'Portuguese',
	#'is': 'Iceland',
	#'ca': 'Canada',
	#'nb': 'Norsk',
	# 'vi': '',
	# 'th': '',
	'en': 'English',
	'de': 'German',
	'es': 'Spanish',
	'fr': 'French',
	'ar': 'Arabic',
	'cn': 'Chinese',
	'ja': 'Japanese',
	'ru': 'Russian',
}

class Movie(object):

	def __init__(self, row):
		self.raw_genres = self.read(row, 'genres')
		self.raw_collection = self.read(row, 'belongs_to_collection')
		self.raw_production_companies = self.read(row, 'production_companies')
		self.raw_cast = self.read(row, 'cast')
		self.raw_crew = self.read(row, 'crew')
		self.raw_keywords = self.read(row, 'Keywords')
		self.raw_production_countries = self.read(row, 'production_countries')
		self.raw_release_date = row['release_date']

		self.original_language = row['original_language']
		self.tagline = row['tagline']
		self.runtime = row['runtime']
		self.popularity = row['popularity']
		self.budget = row['budget']
		self.id = row['id']
		self.title = row['title']
		self.cast = None
		self.budget = row['budget']

		if math.isnan(row['revenue']):
			self.revenue = None
		else:
			self.revenue = row['revenue']

	def read(self, row, key):
		try:
			return ast.literal_eval(row[key])
		except ValueError:
			return None

	def revenue(self):
		if self.raw_revenue.isnan():
			return None
		else:
			return self.raw_revenue

	def release_date(self):
		try:
			m, d, y = self.raw_release_date.split('/')
		except AttributeError:
			print(self.title)		
		
		m, d, y = int(m), int(d), int(y)

		if y <= 19:
			y += 2000
		else:
			y += 1900

		dt = datetime.date(y, m, d)
		zero = datetime.date(1900, 1, 1)
		offset = dt - zero
		return offset.days

	def collection(self):
		try:
			return self.raw_collection[0]['name']
		except TypeError:
			return 'None'

	def in_collection(self):
		if self.collection() == 'None':
			return 0
		else:
			return 1

	def language(self):
		try:
			return language_map[self.original_language]
		except KeyError:
			return 'Other'

	def genres(self):
		try:
			return [genre['name'].lower().replace(' ', '_') for genre in self.raw_genres]
		except TypeError:
			return []

	def director(self):
		for person in self.raw_crew:
			if person['job'] == 'director':
				return person['name']
		return 'None'

	def producers(self):
		return [person['name'] for person in self.raw_crew if person['job'] == 'producer']

	def cast_names(self):
		try:
			return [person['name'] for person in self.raw_cast]
		except TypeError:
			return []

	def star_name(self):
		return self.cast_names()[0]

	def star(self):
		try:
			return self.cast[0]
		except IndexError:
			return None

	def star_average_revenue(self):
		try:
			return (self.star().total_revenue_stared() - self.revenue)/(self.star().num_movies_stared()-1)	
		except AttributeError:
			return 0
		except ZeroDivisionError:
			return 0
		except TypeError:
			return self.star().total_revenue_stared()/self.star().num_movies_stared()

	def star_num_movies(self):
		try:
			return self.star().num_movies()
		except AttributeError:
			return 0

	def average_cast_movies(self):
		try:
			return sum([actor.num_movies() for actor in self.cast])/len(self.cast)
		except ZeroDivisionError:
			return 0

	def top3_cast_movies(self):
		return sum([actor.num_movies() for actor in self.cast[:3]])/3

	def top3_total_revenue(self):
		try:
			return sum([actor.total_revenue() for actor in self.cast[:3]]) - 3*self.revenue
		except TypeError:
			return sum([actor.total_revenue() for actor in self.cast[:3]])

class Actor(object):
	
	def __init__(self, name):
		self.name = name
		self.movies = []

	def num_movies(self):
		return len(self.movies)

	def total_revenue(self):
		return sum(movie.revenue for movie in self.movies if movie.revenue)

	def average_revenue(self):
		return self.total_revenue()/sum([1 for movie in self.movies if movie.revenue])

	def movies_stared(self):
		return [movie for movie in self.movies if movie.star_name() == self.name]

	def num_movies_stared(self):
		return len(self.movies_stared())

	def total_revenue_stared(self):
		return sum(movie.revenue for movie in self.movies_stared() if movie.revenue)

	def average_revenue_stared(self):
		try:
			return self.total_revenue_stared()/sum([1 for movie in self.movies_stared() if movie.revenue])
		except ZeroDivisionError:
			return 0

def build(data):

	actors = {}
	movies = []

	for n, row in data.iterrows():

		movie = Movie(row)
		
		try:
			movie.cast_names()
		except TypeError:
			print(movie.title)		
		
		for actor_name in movie.cast_names():
			try:
				actors[actor_name].movies.append(movie)
			except KeyError:
				actor = Actor(actor_name)
				actor.movies.append(movie)
				actors[actor_name] = actor
			
		movies.append(movie)
	
	for movie in movies:
		movie.cast = [actors[actor_name] for actor_name in movie.cast_names()]

	return movies, to_pandas(movies)

def to_pandas(movies):

	data = []
	for movie in movies:

		_data = {
			'release_date': movie.release_date(),			
			'language': movie.language(),
			'popularity': movie.popularity,
			#'collection': movie.collection(),
			'in_collection': movie.in_collection(),
			'star_average_revenue': movie.star_average_revenue(),
			'star_num_movies': movie.star_num_movies(),
			'cast_average_num_movies': movie.average_cast_movies(),
			'top3_cast_movies': movie.top3_cast_movies(),
			'top3_total_revenue': movie.top3_total_revenue(),
			'budget': movie.budget,
		}

		genres = [
			'adventure', 'animation', 'crime', 'horror',
			'comedy', 'romance', 'drama', 'foreign', 'war', 
			'science_fiction', 'family', 'thriller', 'action',
			'western', 'music', 'history', 'tv_movie',
			'documentary', 'fantasy', 'mystery',
		]

		for genre in genres:
			if genre in movie.genres():
				_data[genre] = 1
			else:
				_data[genre] = 0

		data.append(_data)

	df = pandas.DataFrame(data)
	df = pandas.get_dummies(df)

	return pandas.DataFrame(df)




































