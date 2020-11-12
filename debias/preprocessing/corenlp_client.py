"""A client for a CoreNLP Server."""
import json

import requests


class CoreNLPClient(object):
  """A client that interacts with the CoreNLPServer."""

  def __init__(self, hostname='http://localhost', port=7000):
    """Create the client.
    Args:
      hostname: hostname of server.
      port: port of server.
      cache_file: load and save cache to this file.
    """
    self.hostname = hostname
    self.port = port

  def query(self, sents, properties, sess=None):
    """Most general way to query the server.

    Args:
      sents: Either a string or a list of strings.
      properties: CoreNLP properties to send as part of the request.
    """
    url = '%s:%d' % (self.hostname, self.port)
    params = {'properties': str(properties)}
    if isinstance(sents, list):
      data = '\n'.join(sents)
    else:
      data = sents
    if sess is None:
      r = requests.post(url, params=params, data=data.encode('utf-8'))
    else:
      r = sess.post(url, params=params, data=data.encode('utf-8'))
    r.encoding = 'utf-8'
    try:
      json_response = json.loads(r.text, strict=False)
    except json.JSONDecodeError:
      raise ValueError("CoreNLP error: " + r.text)
    return json_response

  def query_tokenize(self, sents, sess=None):
    """Standard query for getting POS tags."""
    properties = {
      'ssplit.newlineIsSentenceBreak': 'always',
      'annotators': 'tokenize,ssplit',
      'outputFormat': 'json'
    }
    return self.query(sents, properties, sess)

  def query_ner(self, paragraphs, sess=None, whitespace=False):
    """Standard query for getting NERs on raw paragraphs."""
    annotators = 'tokenize,ssplit,pos,ner,entitymentions'
    properties = {
      'ssplit.newlineIsSentenceBreak': 'always',
      'annotators': annotators,
      'outputFormat': 'json'
    }
    if whitespace:
      properties["tokenize.whitespace"] = True
    return self.query(paragraphs, properties, sess)
