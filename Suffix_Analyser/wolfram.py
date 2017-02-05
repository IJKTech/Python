import requests
from xml.etree import ElementTree
import urllib
from database import *
import appid


class WolframAPIClient(object):

	def __init__(self):
		self.databaseManager = DefinitionDatabase()

	def __getDatabaseDefinition(self, word):
		result = self.databaseManager.queryDB('select * from words where word = ?',
                [word], one=True)
		if result != None:
			return result[2]
		else:
			return result

	def __setDatabaseDefinition(self, word, definition):
		result = self.databaseManager.insertDB('words', fields=('word','definition'), values=(word,definition))
		return result

	'''
	Returns parsed XML tree with Wolfram response
	'''
	def __makeWolframQuery(self, query):
		getVars = {'input': query, 'appid': appid.app_id}
		urlBase = 'http://api.wolframalpha.com/v2/query?'
		url = urlBase + urllib.urlencode(getVars)
		
		response = requests.get(url)
		xmlTree = ElementTree.fromstring(response.content)
		return xmlTree

	def __getPlaintextResult(self,tree):
		plaintext = None
		for pod in tree.findall('.//pod'):
			if pod.attrib['title'] == "Result":
				for pt in pod.findall('.//plaintext'):
					if pt.text:
						plaintext = pt.text
		return plaintext

	def __getWordsFromTree(self,tree):
		rawText = self.__getPlaintextResult(tree)
		if rawText != None:
			theWords = rawText.split(" | ")
			return theWords
		else:
			return None

	def __getDefinitionFromTree(self,tree):
		return self.__getPlaintextResult(tree)

	def getWordsFromSuffix(self,suffix):
		wordTree = self.__makeWolframQuery(suffix)
		words = self.__getWordsFromTree(wordTree)
		return words

	def getDefinition(self,word):
		databaseDefinition = self.__getDatabaseDefinition(word)
		if databaseDefinition == None:
			definitionTree = self.__makeWolframQuery("define "+ word)
			wolframDefinition = self.__getDefinitionFromTree(definitionTree)
			if wolframDefinition != None:
				databaseDefinition = wolframDefinition
				self.__setDatabaseDefinition(word, databaseDefinition)
		return databaseDefinition

