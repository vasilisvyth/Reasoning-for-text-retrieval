INSTRUCTION_BETTER_DEM = 'You need to first break the question step by step into subquestions and then combine them to get the original question.'

DEMONSTRATIONS_BETTER_DEM = {0:{
'ex1':'''
Question: 1998 fiction books based on Doctor Who
Let's break down the question step by step:
```
books_1998 = 'find 1998 books'
fiction_books = 'find fiction books'
doctor_who_books = 'find books based on Doctor Who'
```


Combine the subquestions to get the original question:
```
ans = books_1998 and fiction_books and doctor_who_books
```
''',
'ex2':'''
Question: Films based on works by Stanisław Lem
Let's break down the question step by step:
```
stanislaw_lem_films = 'find Films based on works by Stanisław Lem'
```

Combine the subquestions to get the original question:
```
ans = stanislaw_lem_films
```
''',
'ex3':'''
Question: Indian musical and Malayalam films remade in other languages but not featuring an item number
Let's break down the question step by step:
```
indian_musical_films = 'find Indian musical films remade in other languages'
malayalam_films =  'find Malayalam films remade in other languages'
item_number_films = 'find films featuring an item number'
```

Combine the subquestions to get the original question:
```
ans = indian_musical_films and malayalam_films and not item_number_films 
```
''',
'ex4':'''
Question: What are Vultures or Eocene reptiles of South America or Extinct animals of Peru?
Let's break down the question step by step:
```
vultures = 'What are Vultures?'
eocene_reptiles_south_america = 'What are Eocene reptiles of South America?'
extinct_animals_peru = 'What are Extinct animals of Peru?'
```

Combine the subquestions to get the original question:
```
ans = vultures or eocene_reptiles_south_america or extinct_animals_peru
```
''',
'ex5':'''
Question: Birds described in 1991 or Birds of the Western Province (Solomon Islands)
Let's break down the question step by step:
```
birds_1991 = 'find Birds described in 1991'
birds_western_province = 'find Birds of the Western Province (Solomon Islands)'
```

Combine the subquestions to get the original question:
```
ans = birds_1991 or birds_western_province
```
''',
'ex6':'''
Question: New American Library Books about the military
Let's break down the question step by step:
```
new_american_library_books = 'find New American Library Books'
military_books = 'find books about the military'
``` 

Combine the subquestions to get the original question:
```
ans = new_american_library_books and military_books
```
''',
'ex0':'''
Question: Vertebrate animals from Rwanda that are not also Sub-Saharan African mammals
Let's break down the question step by step:
```
vertebrate_animals_rwanda = 'find Vertebrate animals from Rwanda'
sub_Saharan_african_mammals = 'find Sub-Saharan African mammals'
``` 

Combine the subquestions to get the original question:
```
ans = vertebrate_animals_rwanda and not sub_Saharan_african_mammals
```
'''}}
TEST_TEMPLATE_BETTER_DEM = '''
How about this Question?
Question: {question}
Let's break down the question step by step:
'''