import random
def create_rand_demonstrations(seed, num_demonstrations, demonstrations):
    pool_size = len(demonstrations[seed])
    rand_demonstrations_ids = random.sample(range(pool_size), num_demonstrations) # without repetition
    return rand_demonstrations_ids

def concat_demonstations(seed, rand_demonstrations_ids, demonstrations):
    seed_demonstrations = demonstrations[seed]
    demonstations_txt = ''
    for id in rand_demonstrations_ids:
        ex_key = 'ex'+str(id)
        tmp_demonstration_txt = seed_demonstrations[ex_key]
        demonstations_txt += tmp_demonstration_txt
    return demonstations_txt

def concat_test2prompt(demonstations_text, query, test_template):
    demonstations_text += test_template.format(question=query)
    return demonstations_text

# tmp= INSTRUCTION+'\n'+DEMONSTRATIONS[0]['ex1']+DEMONSTRATIONS[0]['ex2']+TEST_TEMPLATE.format(question='This is my question')
# b=1