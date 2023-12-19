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
question_0 = "what are American films"
question_1 = "what are Animated movies"
question_2 = "what are films about superheroes"
question_3 = "what are Movies from the 2010s"

# Step 2. Call the find_docs function
docs_0 = find_docs(question_0) 
docs_1 = find_docs(question_1) 
docs_2 = find_docs(question_2) 
docs_3 = find_docs(question_3) 

# Step 3. Combine the results to get the final answer (ans)
ans = (docs_1 and docs_2 and docs_3) and not docs_0
```
''',
'ex4':'''
Question: Books from 1552 or 1559
```
# Step 1. Break down the question step by step into subquestions
question_0 = "what are 1552 books"
question_1 = "what are 1559 books"

# Step 2. Call the find_docs function
docs_0 = find_docs(question_0)
docs_1 = find_docs(question_1)

# Step 3. Combine the results to get the final answer (ans)
ans = docs_0 or docs_1
```
''',
'ex6':'''
Question: Hungarian thriller films
```
# Step 1. Break down the question step by step into subquestions
question_0 = "what are films from Hungary"
question_1 = "what are thriller films"

# Step 2. Call the find_docs function
docs_0 = find_docs(question_0)
docs_1 = find_docs(question_1)

# Step 3. Combine the results to get the final answer (ans)
ans = docs_0 and docs_1
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
Question: Dutch crime comedy or romantic comedy films
```
# Step 1. Break down the question step by step into subquestions
question_0 = "what are dutch films"
question_1 = "what are crime films"
question_2 = "what are comedy films"
question_3 = "what are romantic films"

# Step 2. Call the find_docs function
docs_0 = find_docs(question_0)
docs_1 = find_docs(question_1)
docs_2 = find_docs(question_2)
docs_3 = find_docs(question_3)

# Step 3. Combine the results to get the final answer (ans)
ans = (docs_0 and docs_1 and docs_2) or (docs_0 and docs_3 and docs_2)
```
'''
}
}
TEST_TEMPLATE_DOCS_NEW = '''
How about this Question?
Question: {question}
'''