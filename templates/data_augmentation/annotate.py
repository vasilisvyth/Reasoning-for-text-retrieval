intersect = """
CONVERT UNION TO INTERSECTION AND SEE WHETHER THE DOCUMENT IS RELEVANT FOR 
THE NEW QUERY Q
INTERSECTION
In toral 5500 annotations are needed

PROMPT: 

The following query is the union of two queries
query: {query}

First, you should write a new query that is the intersection of those simpler queries
Then, you should say whether the following document answers the new query you created

Document: {document} "

Your output must always be a JSON object with the following keys:
 -"new query": string, a new query that is the intersection of those simpler queries
 -"label": int, 1 if the above document answers the new query you created otherwise 0
"""