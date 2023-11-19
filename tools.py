import func_timeout


EXECUTION_TIMEOUT_TIME = 10

def find_docs(query):
    pass

def safe_execute(code_string: str, keys=None):
    '''
    If the keys parameter is a list of variable names, the function attempts to retrieve the values of those specific
    variables from the local variables obtained after executing the code using exec.
    If a variable is not found, its corresponding position in the list will contain None.
    If keys are provided, the function retrieves the values of these variables from the executed code.
    '''
    def execute(x):
        try:
            exec(x)
            locals_ = locals() # create copy of the current local variables
            return locals_
            # if keys is None:
            #     return locals_.get('answer', None)
            # else:
            #     return [locals_.get(k, None) for k in keys]
        except Exception as e:
            print(f'Exception: {e}')
            return None
        
    # If the execution exceeds this time limit raise exception
    try: 
        ans = func_timeout.func_timeout(EXECUTION_TIMEOUT_TIME, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans

def synthesize_program(result: str, prefix: str) -> str:
    program = prefix
    generated_txt = result.split('\n')
    subquestions_block = False
    subquestions = []
    for i, line in enumerate(generated_txt):
        if 'python' in line or line=='':
            continue
        elif line == '```':
            break
        # elif line == '# Define the subquestions':
        #     subquestions_block = True
        # elif line == '# Combine using the correct logical operator if needed':
        #     subquestions_block = False
        # elif subquestions_block:
        #     dict = safe_execute(line)
        #     subquestions.extend(list(dict.values()))


        program += line + '\n'
            
    return program