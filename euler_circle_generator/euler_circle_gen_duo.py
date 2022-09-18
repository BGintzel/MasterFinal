from matplotlib import pyplot as plt

from euler_circle_generator.euler_circle_lib import gen_circle
# from euler_circle_generator.matplotlib_venn._venn2 import venn2
import random
import json
import pickle
import numpy as np



'''
Relational statement format:
Letter:A-Z for different entity
Relationship for Venn-2: 
1. A contains B ['A','>','B']
2. A contained in B ['A','<','B']
3. A intersects B ['A','&','B']
4. A does not intersect B ['A','!','B']
'''
file_path = 'euler_circle_generator/generated_diagram'

relational_operator = ['>', '<', '&', '!']


def area_gen(operator):
    if operator == '>':
        Ab = 100
        aB = 0
        AB = random.uniform(10, Ab)
        return (Ab, aB, AB)
    elif operator == '<':
        Ab = 0
        aB = 100
        AB = random.uniform(10, aB)
        return (Ab, aB, AB)
    elif operator == '&':
        Ab = random.uniform(10, 100)
        aB = random.uniform(10, 100)
        AB = 200 - Ab - aB
        return (Ab, aB, AB)
    elif operator == '!':
        Ab = random.uniform(10, 100)
        aB = random.uniform(10, 100)
        AB = 0
        return (Ab, aB, AB)
    else:
        print('invalid relational operator')


def inference(op1, op2):
    op3 = []
    if op1 == '>':
        if op2 == '>':
            op3 = ['>']
        if op2 == '<':
            op3 = ['<', '>', '&']
        if op2 == '&':
            op3 = ['&', '>']
        if op2 == '!':
            op3 = ['>', '!', '&']
    elif op1 == '<':
        if op2 == '>':
            op3 = ['>', '<', '&', "!"]
        if op2 == '<':
            op3 = ['<']
        if op2 == '&':
            op3 = ['&', '<', '!']
        if op2 == '!':
            op3 = ['!']
    elif op1 == '&':
        if op2 == '>':
            op3 = ['>', '&', '!']
        if op2 == '<':
            op3 = ['<', '&']
        if op2 == '&':
            op3 = ['&', '>', '<', '!']
        if op2 == '!':
            op3 = ['>', '!', '&']
    elif op1 == '!':
        if op2 == '>':
            op3 = ['!']
        if op2 == '<':
            op3 = ['<', '!', '&']
        if op2 == '&':
            op3 = ['&', '!', '<']
        if op2 == '!':
            op3 = ['>', '!', '&', '<']
    label = np.zeros(4)
    for op in op3:
        label[relational_operator.index(op)] = 1
    return op3, label


def create_dataset(num_each_class=1250):
    diag_dict = {}
    img_counter = 0
    euler_record = np.zeros((4 * 4 * num_each_class, 4), dtype=np.uint8)

    size = len(relational_operator)*len(relational_operator)*num_each_class
    counter = 0

    for i in relational_operator:
        for j in relational_operator:
            for t in range(num_each_class):
                counter += 1
                print(f'Creating dataset {int(counter*100/size)}% done')
                diag_id = i + j + '_' + str(t)
                file_prefix = file_path + '/' + str(diag_id)
                op_num_1 = gen_circle(i, file_prefix + '_1.jpg', 'r', 'g')
                op_num_2 = gen_circle(j, file_prefix + '_2.jpg', 'g', 'b')
                relations, label = inference(i, j)
                op_num_3 = np.random.randint(4)
                op_num_4 = np.random.randint(4)
                while op_num_3 == op_num_4:
                    op_num_4 = np.random.randint(3)
                gen_circle(relational_operator[op_num_3], file_prefix + '_3.jpg', 'r', 'b')
                gen_circle(relational_operator[op_num_4], file_prefix + '_4.jpg', 'r', 'b')
                euler_record[img_counter, 0] = op_num_1
                euler_record[img_counter, 1] = op_num_2
                euler_record[img_counter, 2] = label[op_num_3]
                euler_record[img_counter, 3] = label[op_num_4]

                diag_dict[diag_id] = label
                img_counter += 1

    np.savetxt(file_path + '/' + 'euler_record.csv', euler_record.astype(np.uint8), delimiter=',')
    with open(file_path + '/' + 'diag_dict.pickle', 'wb') as fp:
        pickle.dump(diag_dict, fp)

    # venn2(subsets=(10,10,110),set_labels=('K', 'M'))
# plt.show()
