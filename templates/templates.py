template2logic = {
'_ or _':[' or '],
'_ that are not _':[' and ','not '],
'_':[], 
'_ that are also _ but not _': [' and ',' and ','not '],
'_ that are also _':[' and '],
'_ or _ or _':[' or ',' or '],
'_ that are also both _ and _':[' and ',' and ']
}


check_template2logic = {
'_ or _':['or'],
'_ that are not _':['that are not'],
'_':[], 
'_ that are also _ but not _': ['and','not'],
'_ that are also _':['and'],
'_ or _ or _':['or','or'],
'_ that are also both _ and _':['and','and']
}
# Α
# Α or B, A or B or C
# A and B, A and B and C, 
# A and B and not C, A and not B

# Α, Α and B and C, A or B, A and B and not C
# ex2, ex1, ex5, ex3

# I think it will be easier to generalize from 3 operators to 2 operators than the opposite

# Α, A or B or C, A and B, A and B and not C


demonstration_op_map  = {
'ex1':'A and B and C',
'ex2':'A',
'ex3':'A and B and not C',
'ex4':'A or B or C',
'ex5':'A or B',
'ex6':'A and B',
'ex0':'A and not B',
}

INSTRUCTION = 'Think step by step to carry out the Instruction. You are only allowed to write python code.'

DEMONSTRATIONS = {0:{
'ex1':'''
Instruction: 1998 fiction books based on Doctor Who
Program:
```python
question = '1998 fiction books based on Doctor Who'

# Define the subquestions
books_1998 = '1998 books'
fiction_books = 'fiction books'
doctor_who_books = 'books based on Doctor Who'

# Combine using the correct logical operator if needed
answer = books_1998 and fiction_books and doctor_who_books
```
''',
'ex2':'''
Instruction: Films based on works by Stanisław Lem
Program:
```python
question = 'Films based on works by Stanisław Lem'

# Define the subquestions
stanislaw_lem_films = 'Films based on works by Stanisław Lem'

# Combine using the correct logical operator if needed
answer = stanislaw_lem_films
```
''',
'ex3':'''
Instruction: Indian musical and Malayalam films remade in other languages but not featuring an item number
Program:
```python
question = 'Indian musical and Malayalam films remade in other languages but not featuring an item number'

# Define the subquestions
indian_musical_films = 'Indian musical films remade in other languages'
malayalam_films = 'Malayalam films remade in other languages'
item_number_films = 'films featuring an item number'

# Combine using the correct logical operator if needed
answer = indian_musical_films and malayalam_films and not item_number_films
```
''',
'ex4':'''
Instruction: What are Vultures or Eocene reptiles of South America or Extinct animals of Peru?
Program:
```python
question = 'What are Vultures or Eocene reptiles of South America or Extinct animals of Peru?'

# Define the subquestions
vultures = 'What are Vultures?'
eocene_reptiles_south_america = 'What are Eocene reptiles of South America'
extinct_animals_peru = 'What are Extinct animals of Peru?'

# Combine using the correct logical operator if needed
answer = vultures or eocene_reptiles_south_america or extinct_animals_peru
```
''',
'ex5':'''
Instruction: Birds described in 1991 or Birds of the Western Province (Solomon Islands)
Program:
```python
question = 'Birds described in 1991 or Birds of the Western Province (Solomon Islands)'

# Define the subquestions
birds_1991 = 'Birds described in 1991'
birds_western_province = 'Birds of the Western Province (Solomon Islands)'
 
# Combine using the correct logical operator if needed
answer = birds_1991 or birds_western_province
```
''',
'ex6':'''
Instruction: New American Library Books about the military
Program:
```python
question = 'New American Library Books about the military'
 
# Define the subquestions
new_american_library_books = 'New American Library Books'
military_books = 'military books'
 
# Combine using the correct logical operator if needed
answer = new_american_library_books and military_books
```
''',
'ex0':'''
Instruction: Vertebrate animals from Rwanda that are not also Sub-Saharan African mammals
Program:
```python
question = 'Vertebrate animals from Rwanda that are not also Sub-Saharan African mammals'
 
# Define the subquestions
vertebrate_animals_rwanda = 'Vertebrate animals from Rwanda'
sub_Saharan_african_mammals = 'Sub-Saharan African mammals'
 
# Combine using the correct logical operator if needed
answer = vertebrate_animals_rwanda and not sub_Saharan_african_mammals
```
'''}}
TEST_TEMPLATE = '''
How about this Instruction?
Instruction: {question}
Program:
```python
'''

