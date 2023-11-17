query = 'Birds found in the Venezuelan Andes but not in Colombia'

template2logic = {
'_ or _':'A or B',
'_ that are not _':'A and not B',
'_':'A',
'_ that are also _ but not _': 'A and B and not C',
'_ that are also _':'A and B',
'_ or _ or _':'A or B or C',
'_ that are also both _ and _':'A and B and C'
}

#! A and B and C
#! A
#! A and B and not C
#! A or B or C
#! A or B
#! A and B
#! A and not B

templates = {0:'''
Think step by step to carry out the instruction. You are only allowed to write python code.

Instruction: 1998 fiction books based on Doctor Who
Program:
```
question = '1998 fiction books based on Doctor Who'

# Define the subquestions
books_1998 = '1998 books'
fiction_books = 'fiction books'
doctor_who_books = 'books based on Doctor Who'

# Combine using the correct logical operator if needed
answer = books_1998 and fiction_books and doctor_who_books
```
Instruction: Films based on works by Stanisław Lem
Program:
```
question = 'Films based on works by Stanisław Lem'

# Define the subquestions
stanislaw_lem_films = 'Films based on works by Stanisław Lem'

# Combine using the correct logical operator if needed
answer = stanislaw_lem_films
```
 
Instruction: Indian musical and Malayalam films remade in other languages but not featuring an item number
Program:
```
question = 'Indian musical and Malayalam films remade in other languages but not featuring an item number'

# Define the subquestions
indian_musical_films = 'Indian musical remade in other languages'
malayalam_films = 'Malayalam films remade in other languages'
item_number_films = 'films featuring an item number'

# Combine using the correct logical operator if needed
answer = indian_musical_films and malayalam_films and not item_number_films
```
 
Instruction: What are Vultures or Eocene reptiles of South America or Extinct animals of Peru?
Program:
```
question = 'What are Vultures or Eocene reptiles of South America or Extinct animals of Peru?'

# Define the subquestions
vultures = 'What are Vultures?'
eocene_reptiles_south_america = 'What are Eocene reptiles of South America'
extinct_animals_peru = 'What are Extinct animals of Peru?'

# Combine using the correct logical operator if needed
answer = vultures or eocene_reptiles_south_america or extinct_animals_peru
```
 
Instruction: Birds described in 1991 or Birds of the Western Province (Solomon Islands)
Program:
```
question = 'Birds described in 1991 or Birds of the Western Province (Solomon Islands)'

# Define the subquestions
birds_1991 = 'Birds described in 1991'
birds_western_province = 'Birds of the Western Province (Solomon Islands)'
 
# Combine using the correct logical operator if needed
answer = birds_1991 or birds_western_province
```
 
Instruction: New American Library Books about the military
Program:
```
question = 'New American Library Books about the military'
 
# Define the subquestions
new_american_library_books = 'New American Library Books'
military_books = 'military books'
 
# Combine using the correct logical operator if needed
answer = new_american_library_books and military_books
```

Instruction: Vertebrate animals from Rwanda that are not also Sub-Saharan African mammals
Program:
```
question = 'Vertebrate animals from Rwanda that are not also Sub-Saharan African mammals'
 
# Define the subquestions
vertebrate_animals_rwanda = 'Vertebrate animals from Rwanda'
sub_Saharan_african_mammals = 'Sub-Saharan African mammals'
 
# Combine using the correct logical operator if needed
answer = vertebrate_animals_rwanda and not sub_Saharan_african_mammals
```
'''}

tmp= templates[0]
b=1