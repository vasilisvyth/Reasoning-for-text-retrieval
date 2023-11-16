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

{0:'''
Think step by step to carry out the instruction. You are only allowed to write python code.

Instruction: 1998 fiction books based on Doctor Who
Program:
question = 1998 fiction books based on Doctor Who

# Define the subquestions
books_1998 = '1998 books'
fiction_books = 'Fiction books'
doctor_who_books = 'Books based on Doctor Who'

# Find the relevant docs for each subquestion
docs_1998 = find_docs(books_1998)
docs_fiction = find_docs(fiction_books)
docs_doctor_who  = find_docs(doctor_who_books) 

# Combine using the correct logical operator if needed
answer = docs_1998 and docs_fiction and docs_doctor_who

Instruction: Films based on works by Stanisław Lem
Program:
question = 'Films based on works by Stanisław Lem'

# Define the subquestions
stanislaw_lem_films = 'Films based on works by Stanisław Lem'

# Find the relevant docs for each subquestion
docs_stanislaw_lem  = find_docs(stanislaw_lem_films) 

# Combine using the correct logical operator if needed
answer = docs_stanislaw_lem

Instruction: Indian musical and Malayalam films remade in other languages but not featuring an item number
Program:
question = 'Indian musical and Malayalam films remade in other languages but not featuring an item number'

# Define the subquestions
indian_musical_films = 'Indian musical remade in other languages'
malayalam_films = 'Malayalam films remade in other languages'
item_number_films = 'Films featuring an item number'

# Find the relevant docs for each subquestion
docs_indian_musical = find_docs(indian_musical_films)
docs_malayalam = find_docs(malayalam_films)
docs_item_number = find_docs(item_number_films)

# Combine using the correct logical operator if needed
answer = docs_indian_musical and docs_malayalam and not docs_item_number
 
Instruction: What are Vultures or Eocene reptiles of South America or Extinct animals of Peru?
Program:
question = 'What are Vultures or Eocene reptiles of South America or Extinct animals of Peru?'

# Define the subquestions
vultures = 'What are Vultures?'
eocene_reptiles_south_america = 'What are Eocene reptiles of South America'
extinct_animals_peru = 'What are Extinct animals of Peru?'

# Find the relevant docs for each subquestion
docs_vultures = find_docs(vultures)
docs_eocene_reptiles_south_america = find_docs(eocene_reptiles_south_america)
docs_extinct_animals_peru = find_docs(extinct_animals_peru)

# Combine using the correct logical operator if needed
answer = docs_vultures or docs_eocene_reptiles_south_america or docs_extinct_animals_peru

Instruction: Birds described in 1991 or Birds of the Western Province (Solomon Islands)
Program:
question = 'Birds described in 1991 or Birds of the Western Province (Solomon Islands)'

# Define the subquestions
birds_1991 = 'Birds described in 1991'
birds_western_province = 'Birds of the Western Province (Solomon Islands)'

# Find the relevant docs for each subquestion 
docs_birds_1991 = find_docs(birds_1991)
docs_birds_western_province = find_docs(birds_western_province)
 
# Combine using the correct logical operator if needed
answer = docs_birds_1991 or docs_birds_western_province

Instruction: New American Library Books about the military
Program:
question = 'New American Library Books about the military'
 
# Define the subquestions
new_american_library_books = 'New American Library Books'
military_books = 'military books'
 
# Find the relevant docs for each subquestion 
docs_new_american_library_books = find_docs(new_american_library_books)
docs_military_books = find_docs(military_books)

# Combine using the correct logical operator if needed
answer = docs_new_american_library_books and docs_military_books

Instruction: Vertebrate animals from Rwanda that are not also Sub-Saharan African mammals
Program:
question = 'Vertebrate animals from Rwanda that are not also Sub-Saharan African mammals'
 
# Define the subquestions
vertebrate_animals_rwanda = 'Vertebrate animals from Rwanda'
sub_Saharan_african_mammals = 'Sub-Saharan African mammals'
 
# Find the relevant docs for each subquestion 
docs_vertebrate_rwanda = find_docs(vertebrate_animals_rwanda)
docs_sub_saharan_african = find_docs(sub_Saharan_african_mammals)

# Combine using the correct logical operator if needed
answer = docs_vertebrate_rwanda and not docs_sub_saharan_african

'''}