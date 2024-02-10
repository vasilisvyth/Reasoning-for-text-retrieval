# clear, understandable with some effort, ambiguous
# query_type {extremely long-tail, long-tail, common}, “{query_length}” ∈ {less than 5 words, 5 to 15 words, at least 10 words
# Fix the following error
# File "c:\Users\vasil\reasoning-for-text-retrieval\Reasoning-for-text-retrieval\templates\data_augmentation\synthetic_data.py", line 23, in <module>
#     instruct = instructions[0].format(domain='films')
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# KeyError: 'title'

instructions =[ \
'''
The JSON object must contain the following keys:
- "user_query": a string, a random user search query which is the intersection of 2 simpler queries query_A and query_B.
- "query_A": a string, a simpler query that when combined with query_B it gives us the user_query
- "query_B": a string, a simpler query that when combined with query_A  it gives us the user_query
- "positive_document": {{'title': a string, representing a document title of a relevant {domain}, "text": a string, a long relevant specific Wikipedia {domain} page for the user query.}}
- "hard_negative_document": {{'title': a string, representing a document title of a hard negative {domain}, "text": a string, a long Wikipedia {domain} page that only answers either query_A or query_B but not both query_A and query_B!}}
Please adhere to the following guidelines:
- The "user_query" should be around 10 words and it should be about {domain}s.
- Do not provide any explanation in the "text" of the "hard_negative_document" about why it is not relevant to the query.
- No comparison is needed between the text of the "positive_document" and the "hard_negative_document"
- The "text" in each of "positive_document" and "hard_negative_document" should be at least 350 words long and it should be in English.
Your output must always be a JSON object only, do NOT explain yourself or output anything else. Be creative!
''',
'''
The JSON object must contain the following keys:
- "user_query": a string, a random user search query formed by the intersection (AND) of two simpler queries, query_A and query_B.
- "query_A": A string representing one of the simpler queries that, when combined with query_B, forms the user_query.
- "query_B": A string representing the other simpler query that, when combined with query_A, forms the user_query.
- "positive_document": A dictionary with keys "title" and "text", where:
    - "title": A string representing the title of a relevant {domain} document.
    - "text": A long, English text extracted from a specific Wikipedia page related to the user query. It must be at least 350 words in length.
- "hard_negative_document": A dictionary with keys "title" and "text", where:
    - "title": A string representing the title of a hard negative {domain} document.
    - "text": A long, English text extracted from a Wikipedia page related to {domain}s, which only answers either query_A or query_B but not both. It must be at least 350 words in length.
Please also adhere to the following guidelines:
- The "user_query" should be around 10 words and it should be about {domain}s.
- No explanation is needed in the "text" of the "hard_negative_document" about why it is not relevant to the query.
- No comparison is needed between the text of the "positive_document" and the "hard_negative_document"
Your output must always be a JSON object only, do NOT explain yourself or output anything else. Be creative!
''',
# changed the order of queries and removed hard negative text that claims 'it should answer only query_A or query_B but not both'
'''
The JSON object must contain the following keys:
- "user_query": a string, a random user search query formed by the intersection (AND) of two simpler queries.
- "positive_document": A dictionary with keys "title" and "text", where:
    - "title": A string representing the title of a relevant {domain} document.
    - "text": A long, English text extracted from a specific Wikipedia page related to the user query. It must be at least 350 words in length.
- "hard_negative_document": A dictionary with keys "title" and "text", where:
    - "title": A string representing the title of a hard negative {domain} document.
    - "text": A long, English text extracted from a Wikipedia page related to {domain}s that only appears relevant to the query. It must be at least 350 words in length.
Please also adhere to the following guidelines:
- The "user_query" should be around 10 words and it should be about {domain}s.
- Avoid substantial word overlaps between the "user_query" and the "text" of the positive document, otherwise the task would be too easy.
- No explanation should be provided in the "text" of the "hard_negative_document" about why it is not relevant to the query.
- No comparison is needed between the text of the "positive_document" and the "hard_negative_document"
Your output must always be a JSON object only, do NOT explain yourself or output anything else. Be creative!
'''
]
