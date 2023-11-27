'''
the only difference with templates_better_dem is that we 
use question_i and find_docs
'''

INSTRUCTION_BETTER_DEM = 'You need to first break the question step by step into subquestions and then combine them to get the answer to the original question.'


DEMONSTRATIONS_BETTER_DEM = {0:{
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
question_0 = "find Films based on works by Stanisław Lem"
docs_0 = find_docs(question_0) 
```

Combine them to get the answer to the original question:
```
ans = docs_0
```
''',
'ex3':'''
Question: Indian musical and Malayalam films remade in other languages but not featuring an item number
Let's break down the question step by step:
```
question_0 = "find Indian musical films remade in other languages"
docs_0 = find_docs(question_0) 

question_1 =  "find Malayalam films remade in other languages"
docs_1 = find_docs(question_1) 

question_2 = "find films featuring an item number"
docs_2 = find_docs(question_2) 
```

Combine them to get the answer to the original question:
```
ans = docs_0 and docs_1 and not docs_2 
```
''',
'ex4':'''
Question: What are Vultures or Eocene reptiles of South America or Extinct animals of Peru?
Let's break down the question step by step:
```
question_0 = "What are Vultures?"
docs_0 = find_docs(question_0) 

question_1 = "What are Eocene reptiles of South America?"
docs_1 = find_docs(question_1) 

question_2 = "What are Extinct animals of Peru?"
docs_2 = find_docs(question_2) 
```

Combine them to get the answer to the original question:
```
ans = docs_0 or docs_1 or docs_2
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
Question: New American Library Books about the military
Let's break down the question step by step:
```
question_0 = "find New American Library Books"
docs_0 = find_docs(question_0)

question_1 = "find books about the military"
docs_1 = find_docs(question_1)
``` 

Combine them to get the answer to the original question:
```
ans = docs_0 and docs_1
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
TEST_TEMPLATE_BETTER_DEM = '''
How about this Question?
Question: {question}
Let's break down the question step by step:
'''