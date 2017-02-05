
function doQuery()
{
	var suffix = $("#query").val();
	$("#noresults").hide();
	$("#results").empty();
	$("#results").append("<li id='loading'>Loading...</li>");
	
  var search = "words ending in " + suffix;
  var url = encodeURI("wolfram-query?query="+search);
  
  var wolframStream = new EventSource(url);
  
  wolframStream.onmessage = function (e) {
    $("#loading").hide();
    
    var rawString = e.data;
    var parsedMessage = JSON.parse(rawString);
    $("#results").append("<li>"+parsedMessage["word"] + ": " + parsedMessage["definition"] + "</li>");
    
    if (parsedMessage["last"])
    {
      wolframStream.close();
    }
  };

  wolframStream.onerror = function(e) {
    wolframStream.close();
  };

  wolframStream.onend = function(e) {
    wolframStream.close();
  };

}