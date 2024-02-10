Instruction = ['Given a web search query, retrieve relevant passages that answer the query',
    "Given a web search query, retrieve relevant passages that answer the query. If the query is the intersection (and) of subqueries a passage should be relevant to all subquestions.",
    "Given a web search query, retrieve relevant passages that answer the query. Let's think step by step",
]



templates_dict = {
# A and B
'_ that are also _':[
    'Given web search queries, retrieve relevant passages that answer all of the following queries',
    'Given web search queries, retrieve relevant passages that every one of the following queries'
],
# A and B and C
'_ that are also both _ and _':[
    'Given web search queries, retrieve relevant passages that answer all of the following queries',
    'Given web search queries, retrieve relevant passages that every one of the following queries'
],

# A or B
'_ or _': [
    'Given web search queries, retrieve relevant passages that answer at least one of the following queries',
    'Given web search queries, retrieve relevant passages that answer any one of the following queries'
],
# A or B or C
'_ or _ or _': [
    'Given web search queries, retrieve relevant passages that answer at least one of the following queries',
    'Given web search queries, retrieve relevant passages that answer any one of the following queries'
],

# A difference B
'_ that are not _': [
    '''Given web search queries, retrieve relevant passages that answer query 1 but do not answer query 2''',
    '''Given web search queries, retrieve relevant passages that answer query 1 but exclude passages that answer query 2'''
],
# A and B difference C
'_ that are also _ but not _': [
    '''Given web search queries, retrieve relevant passages that answer both query 1 and query 2 but do not answer query 3''',
    '''Given web search queries, retrieve relevant passages that answer both query 1 and query 2 but exclude passages answer query 3'''
],
# A
'_':[
    'Given a web search query, retrieve relevant passages that answer the query'
]

}