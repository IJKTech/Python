from flask import Flask, render_template, Response
from flask import request
from flask import jsonify
from wolfram import *
import appid
import json

app = Flask(__name__)

@app.route("/")
def template():
	return render_template('template.html')

def generateDefinitions(wolframClient,words):
	with app.app_context():
		if words != None:
			for word in words[:appid.limit]:
				definition = wolframClient.getDefinition(word)
				if definition == None:
					definition = "Could not find definition"
				theJSON = {"word":word, "definition":definition, "last":word==words[appid.limit-1]}
				yield "data:"+json.dumps(theJSON)+"\n\n"

@app.route('/wolfram-query', methods=['GET'])
def wolframQuery():
	wolframClient = WolframAPIClient()
	
	suffix_text = request.args.get('query')
	words = wolframClient.getWordsFromSuffix(suffix_text)

	return Response(generateDefinitions(wolframClient,words), mimetype='text/event-stream')


if __name__ == '__main__':
	app.run(debug=True)
