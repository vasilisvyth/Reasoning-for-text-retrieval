doc2query = '''
I will give you a document and you should first write a few characteristics of this specific {domain} based on the given document.
Then, you should create a query which combines 2 of those characteristics using words that express the combination (e.g. 'both', 'and') 
The query should not explicitly refer to the {domain}! The query should be general maybe other {domain}s are relevant to it. The query should be at most 10 words!
The JSON object must contain the following keys:
    'characteristics': string, a few important keywords (characteristics) of the given {domain}
    'query': string, a question that combines few keywords from the 'characteristics' in order to find {domain}s that have these keywords

Here is the {domain} document:
{document}

Your output must always be a JSON object only, do NOT explain yourself or output anything else. Be creative!
'''

# ChatGPT Generated
{
  "characteristics": "epic crime film, Corleone family, transformation, mafia boss",
  "query": "Which films feature both a crime family saga and a character transformation?"
}
