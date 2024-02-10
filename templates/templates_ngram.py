INSTRUCTION_DOCS_N_GRAM = 'You are an expert python programmer. You need to first break the question step by step into subquestions and then combine them to get the answer to the original question.  You are only allowed to write python code.'


DEMONSTRATIONS_DOCS_N_GRAM = {0:{
'ex1':'''
Question: 1998 fiction books based on Doctor Who
Let's break down the question step by step:
```
question_0 = "find 1998 books"
docs_0 = find_docs(question_0) 

question_1 = "find fiction books"
docs_1 = find_docs(question_1) 

question_2 = "find books based on Doctor Who"
docs_2 = find_docs(question_2) 
```

Combine them to get the answer to the original question:
```
ans = docs_0 and docs_1 and docs_2
```
''',
'ex2':'''
Question: Films based on works by Stanisław Lem
Let's break down the question step by step:
```
question_0 = "Films based on works by Stanisław Lem"
docs_0 = find_docs(question_0) 
```

Combine them to get the answer to the original question:
```
ans = docs_0
```
''',
'ex3':'''
Question: Flora of Arizona, also of Oregon, but not of the Sierra Nevada (United States)
Let's break down the question step by step:
```
question_0 = "Flora of Arizona"
docs_0 = find_docs(question_0) 

question_1 =  "Oregon"
docs_1 = find_docs(question_1) 

question_2 = "Sierra Nevada (United States)"
docs_2 = find_docs(question_2) 
```

Combine them to get the answer to the original question:
```
ans = docs_0.intersection(docs_1).difference(docs_2) 
```
''',
'ex4':'''
Question: What are Vultures or Eocene reptiles of South America or Extinct animals of Peru?
Let's break down the question step by step:
```
question_0 = "What are Vultures?"
docs_0 = find_docs(question_0) 

question_1 = "Eocene reptiles of South America?"
docs_1 = find_docs(question_1) 

question_2 = "Extinct animals of Peru?"
docs_2 = find_docs(question_2) 
```

Combine them to get the answer to the original question:
```
ans = docs_0.union(docs_1).union(docs_2)
```
''',
'ex5':'''
Question: Birds described in 1991 or Birds of the Western Province (Solomon Islands)
Let's break down the question step by step:
```
question_0 = "find Birds described in 1991"
docs_0 = find_docs(question_0)

question_1 = "find Birds of the Western Province (Solomon Islands)"
docs_1 = find_docs(question_1)
```

Combine them to get the answer to the original question:
```
ans = docs_0 or docs_1
```
''',
'ex6':'''
Question: 1979 comedy films about the arts
Let's break down the question step by step:
```
question_0 = "1979 comedy films"
docs_0 = find_docs(question_0)

question_1 = "about the arts"
docs_1 = find_docs(question_1)
``` 

Combine them to get the answer to the original question:
```
ans = docs_0.intersection(docs_1)
```
''',
'ex0':'''
Question: Vertebrate animals from Rwanda that are not also Sub-Saharan African mammals
Let's break down the question step by step:
```
question_0 = "find Vertebrate animals from Rwanda"
docs_0 = find_docs(question_0)

question_1 = "find Sub-Saharan African mammals"
docs_1 = find_docs(question_1)
``` 

Combine them to get the answer to the original question:
```
ans = docs_0 and not docs_1
```
'''}}
TEST_TEMPLATE_DOCS_N_GRAM = '''
How about this Question?
Question: {question}
Let's break down the question step by step:
'''