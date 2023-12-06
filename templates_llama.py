INSTRUCTION_DOCS_ANON = 'You need to first break the question step by step into subquestions and then combine them to get the answer to the original question.'
#Instruction:\n{instruction}\n\n### Response:

DEMONSTRATIONS_DOCS_ANON = {0:{
'ex1':'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n1998 fiction books based on Doctor Who

### Response:
question_0 = "find 1998 books"
docs_0 = find_docs(question_0) 
question_1 = "find fiction books"
docs_1 = find_docs(question_1) 
question_2 = "find books based on Doctor Who"
docs_2 = find_docs(question_2) 
ans = docs_0 and docs_1 and docs_2

''',
'ex2':'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nFilms based on works by Stanisław Lem

### Response:
question_0 = "find Films based on works by Stanisław Lem"
docs_0 = find_docs(question_0) 
ans = docs_0

''',
'ex3':'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nIndian musical and Malayalam films remade in other languages but not featuring an item number

### Response:
question_0 = "find Indian musical films remade in other languages"
docs_0 = find_docs(question_0) 
question_1 =  "find Malayalam films remade in other languages"
docs_1 = find_docs(question_1) 
question_2 = "find films featuring an item number"
docs_2 = find_docs(question_2) 
ans = docs_0 and docs_1 and not docs_2 

''',
'ex4':'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat are Vultures or Eocene reptiles of South America or Extinct animals of Peru?

### Response:
question_0 = "What are Vultures?"
docs_0 = find_docs(question_0) 
question_1 = "What are Eocene reptiles of South America?"
docs_1 = find_docs(question_1) 
question_2 = "What are Extinct animals of Peru?"
docs_2 = find_docs(question_2) 
ans = docs_0 or docs_1 or docs_2

''',
'ex5':'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nBirds described in 1991 or Birds of the Western Province (Solomon Islands)

### Response:
question_0 = "find Birds described in 1991"
docs_0 = find_docs(question_0)
question_1 = "find Birds of the Western Province (Solomon Islands)"
docs_1 = find_docs(question_1)
ans = docs_0 or docs_1

''',
'ex6':'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nNew American Library Books about the military

### Response:
question_0 = "find New American Library Books"
docs_0 = find_docs(question_0)
question_1 = "find books about the military"
docs_1 = find_docs(question_1)
ans = docs_0 and docs_1

''',
'ex0':'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nVertebrate animals from Rwanda that are not also Sub-Saharan African mammals

### Response:
question_0 = "find Vertebrate animals from Rwanda"
docs_0 = find_docs(question_0)
question_1 = "find Sub-Saharan African mammals"
docs_1 = find_docs(question_1)
ans = docs_0 and not docs_1

'''}}
TEST_TEMPLATE_DOCS_ANON = '''
Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}

### Response:
'''