from python_interpreter import find_logical_operators

def test_logical_operators():
    program = 'la or '
    sorted_ops_in_program = find_logical_operators(program)
    assert(sorted_ops_in_program == [' or '])
    program =' la or  not '
    sorted_ops_in_program = find_logical_operators(program)
    assert(sorted_ops_in_program == [' or ','not '])
    program ='my name and andyour '
    sorted_ops_in_program = find_logical_operators(program)
    assert(sorted_ops_in_program == [' and '])
    program = ' and  not dew'
    sorted_ops_in_program = find_logical_operators(program)
    assert(sorted_ops_in_program == [' and ','not '])
    program = ' and  not al and la'
    sorted_ops_in_program = find_logical_operators(program)
    assert(sorted_ops_in_program == [' and ','not ',' and '])
    a=1

test_logical_operators()
    