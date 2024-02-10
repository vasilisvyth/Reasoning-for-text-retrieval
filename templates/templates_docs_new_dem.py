'''
the only difference with templates_better_dem is that we 
use question_i and find_docs
'''

INSTRUCTION_DOCS_NEW = 'You are an expert python programmer. You need to first break the question step by step into subquestions and then combine them to get the answer to the original question.  Ensure that the generated subquestions are concise and make sense. You are only allowed to write python code.'

#! A and B and C and not C, #! A or B, #! A and B, #! A
DEMONSTRATIONS_DOCS_NEW = {0:{
'ex2':'''
Question: Non-American animated superhero films from the 2010s
```
# Step 1. Break down the question step by step into subquestions
question_0 = "what are some American films"
question_1 = "what are some Animated films about superheroes"
question_2 = "what are some Movies from the 2010s"

# Step 2. Call the find_docs function
docs_0 = find_docs(question_0) 
docs_1 = find_docs(question_1) 
docs_2 = find_docs(question_2) 

# Step 3. Combine the results to get the final answer (ans)
ans = docs_1.intersection(docs_2).difference(docs_0)
```
''',
'ex4':'''
Question: Books from 1552 or 1559
```
# Step 1. Break down the question step by step into subquestions
question_0 = "what are some 1552 books"
question_1 = "what are some 1559 books"

# Step 2. Call the find_docs function
docs_0 = find_docs(question_0)
docs_1 = find_docs(question_1)

# Step 3. Combine the results to get the final answer (ans)
ans = docs_0.union(docs_1)
```
''',
'ex6':'''
Question: Hungarian thriller films
```
# Step 1. Break down the question step by step into subquestions
question_0 = "what are some films from Hungary"
question_1 = "what are some thriller films"

# Step 2. Call the find_docs function
docs_0 = find_docs(question_0)
docs_1 = find_docs(question_1)

# Step 3. Combine the results to get the final answer (ans)
ans = docs_0.intersection(docs_1)
```
''',
'ex3':'''
Question: Crustaceans of Japan
```
# Step 1. Break down the question step by step into subquestions
question_0 = "what are Crustaceans of Japan"

# Step 2. Call the find_docs function
docs_0 = find_docs(question_0)

# Step 3. Combine the results to get the final answer (ans)
ans = docs_0
```
''',
'ex1':'''
Question: Amphibians of Cambodia but not of Laos
```
# Step 1. Break down the question step by step into subquestions
question_0 = "what are Amphibians of Cambodia"
question_1 = "what are Amphibians of Laos"

# Step 2. Call the find_docs function
docs_0 = find_docs(question_0)
docs_1 = find_docs(question_1)

# Step 3. Combine the results to get the final answer (ans)
ans = docs_0.difference(docs_1)
```
''',
'ex5':'''
Question: Trees of mexico and central america also Monotypic eudicot genera
```
# Step 1. Break down the question step by step into subquestions
question_0 = "what are some trees of mexico"
question_1 = "what are some trees of central america"
question_2 = "what are some Monotypic eudicot genera"

# Step 2. Call the find_docs function
docs_0 = find_docs(question_0)
docs_1 = find_docs(question_1)
docs_2 = find_docs(question_2)

# Step 3. Combine the results to get the final answer (ans)
ans = docs_0.intersection(docs_1).intersection(docs_2)
```
'''

}
}
TEST_TEMPLATE_DOCS_NEW = '''
How about this Question?
Question: {question}
'''