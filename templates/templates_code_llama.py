INSTRUCTION_USER_ASSISTANT ={'role':'system','content':'Write python code that first breaks the instruction step by step into subquestions and then combine them to get the answer to the original question.\
                             You can write find_docs to retrieve the relevant documents.'
}
DEMONSTRATIONS_USER_ASSISTANT = {0:{
'ex1':[{'role':'user','content':'Instruction:\n1998 fiction books based on Doctor Who'},
{'role':'assistant','content':"""
question_0 = "find 1998 books"
docs_0 = find_docs(question_0) 
question_1 = "find fiction books"
docs_1 = find_docs(question_1) 
question_2 = "find books based on Doctor Who"
docs_2 = find_docs(question_2) 
ans = docs_0 and docs_1 and docs_2
"""}],
'ex2':[{'role':'user','content':'Instruction:\nFilms based on works by Stanisław Lem'},
{'role':'assistant','content':"""   
question_0 = "find Films based on works by Stanisław Lem"
docs_0 = find_docs(question_0) 
ans = docs_0
"""}],
'ex3':[{'role':'user','content':'Instruction:\nIndian musical and Malayalam films remade in other languages but not featuring an item number'},
{'role':'assistant','content':"""  
question_0 = "find Indian musical films remade in other languages"
docs_0 = find_docs(question_0) 
question_1 =  "find Malayalam films remade in other languages"
docs_1 = find_docs(question_1) 
question_2 = "find films featuring an item number"
docs_2 = find_docs(question_2) 
ans = docs_0 and docs_1 and not docs_2 
"""}],
'ex4':[{'role':'user','content':'Instruction:\nWhat are Vultures or Eocene reptiles of South America or Extinct animals of Peru?'},
{'role':'assistant','content':"""  
question_0 = "What are Vultures?"
docs_0 = find_docs(question_0) 
question_1 = "What are Eocene reptiles of South America?"
docs_1 = find_docs(question_1) 
question_2 = "What are Extinct animals of Peru?"
docs_2 = find_docs(question_2) 
ans = docs_0 or docs_1 or docs_2
"""}],
'ex5':[{'role':'user','content':'Instruction:\nBirds described in 1991 or Birds of the Western Province (Solomon Islands)'},
{'role':'assistant','content':"""
question_0 = "find Birds described in 1991"
docs_0 = find_docs(question_0)
question_1 = "find Birds of the Western Province (Solomon Islands)"
docs_1 = find_docs(question_1)
ans = docs_0 or docs_1
"""}],
'ex6':[{'role':'user','content':'Instruction:\nNew American Library Books about the military'},
{'role':'assistant','content':"""
question_0 = "find New American Library Books"
docs_0 = find_docs(question_0)
question_1 = "find books about the military"
docs_1 = find_docs(question_1)
ans = docs_0 and docs_1
"""}],
'ex0':[{'role':'user','content':'Instruction:\nVertebrate animals from Rwanda that are not also Sub-Saharan African mammals'},
{'role':'assistant','content':"""
question_0 = "find Vertebrate animals from Rwanda"
docs_0 = find_docs(question_0)
question_1 = "find Sub-Saharan African mammals"
docs_1 = find_docs(question_1)
ans = docs_0 and not docs_1
"""}]}}

TEST_TEMPLATE_USER_ASSISTANT = """Write python code that first breaks the instruction step by step into subquestions and then combine them to get the answer to the original question.
Instruction:\n{question}"""