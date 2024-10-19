import csv
import numpy as np


def preprocess(sample):
    pp_sample = sample.copy()

    # sort Age into buckets -- TODO: maximize information gained from splits
    # age ranges from 15 - 74, let's split into 15-24, 25-34, ..., 65-74
    age_i = attributes.index('Age')
    pp_sample[age_i] = (float(pp_sample[age_i]) - 15) // 10

    # sort Na_to_K to buckets -- TODO: maximize information gained from splits
    na_i = attributes.index('Na_to_K')
    pp_sample[na_i] = (float(pp_sample[na_i]) - 6.269) // 5

    return pp_sample


def plurality_value(examples):
    counts = {}
    for example in examples:
        label = example[-1]
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    max_label = None
    max_value = 0
    for label, value in counts.items():
        if value > max_value:
            max_value = value
            max_label = label
    return max_label


def all_examples_have_same_labels(examples):
    set_labels = set(example[-1] for example in examples)
    if len(set_labels) == 1:
        return True
    return False


def get_most_important_A(attributes, examples):
    # most important is the smallest remainder (no need to calculate H(root) or gain)
    # since H(root) is the same for each A selected

    # get remainder for each A
    min_remainder = np.inf
    min_remainder_i = None

    for attribute_i in range(len(attributes)-1):
        remainder = 0
        attr_vals = list(set(ex[attribute_i] for ex in examples))
        # For each get subset E where the subset has single value for attr
        for attr_val in attr_vals:
            E_subset = [ex for ex in examples if ex[attribute_i] == attr_val]
            num_ex_in_branch = len(E_subset)
            num_ex_in_parent = len(examples)
            P_branch = num_ex_in_branch / num_ex_in_parent

            entropy = 0
            drugs = list(set(ex[-1] for ex in E_subset))
            for drug in drugs:
                # get the probability of each drug in E_subset
                num_drug = sum([1 for ex in E_subset if ex[-1] == drug])
                P_drug = num_drug / len(E_subset)
                entropy += -1 * P_drug * np.log2(P_drug)

            remainder += P_branch * entropy

        if remainder < min_remainder:
            min_remainder = remainder
            min_remainder_i = attribute_i

    return attributes[min_remainder_i], min_remainder_i


class Node:
    def __init__(self, parent_node, test_attr, examples, is_leaf=False, pred=None):
        self.parent_node = parent_node
        self.test_attr = test_attr
        self.examples = examples
        self.children_nodes = {}
        self.is_leaf = is_leaf
        self.pred = pred

    def add_child(self, subtree, vk):
        self.children_nodes[vk] = subtree


def decision_tree_learning(examples, attributes, parent_node):
    # base case 1: examples is empty, return plurality value of parents
    if len(examples) == 0:
        pred = plurality_value(parent_node.examples)
        return Node(parent_node, None, examples, is_leaf=True, pred=pred)

    # base case 2: all examples have the same classification
    elif all_examples_have_same_labels(examples):
        pred = examples[0][-1]
        return Node(parent_node, None, examples, is_leaf=True, pred=pred)

    # base case 3: attributes is empy, return plurality value of examples
    elif len(attributes) == 1:  # last attribute is the classification
        pred = plurality_value(examples)
        return Node(parent_node, None, examples, is_leaf=True, pred=pred)

    # expand tree
    else:
        A, A_i = get_most_important_A(attributes, examples)
        remaining_attrs = [a for a in attributes if a != A]
        tree = Node(parent_node, A, examples)

        vk_s = list(set(ex[A_i] for ex in examples))
        for vk in vk_s:
            exs = [ex[:A_i] + ex[A_i+1:] for ex in examples if ex[A_i] == vk]
            subtree = decision_tree_learning(exs, remaining_attrs, tree)
            tree.add_child(subtree, vk)

    return tree


def forward(pp_sample, dt):
    # base case: dt is a classification
    if dt.is_leaf == True:
        return dt.pred

    # otherwise, recurse down tree
    test_attr = dt.test_attr
    test_attr_i = attributes.index(test_attr)
    samp_attr_v = pp_sample[test_attr_i]
    sub_dt = dt.children_nodes[samp_attr_v]
    return forward(pp_sample, sub_dt)


# import data -- get examples, attributes
with open("drug200.csv", 'r') as f_csv:
    csv_reader = csv.reader(f_csv)

    # get attributes
    attributes = next(csv_reader)

    # get examples
    examples = []
    for example in csv_reader:
        examples.append(example)

# preprocess
pp_examples = [preprocess(ex) for ex in examples]

# call function
dt = decision_tree_learning(pp_examples, attributes, None)

# get classification for new sample
sample = ['27', 'M', 'NORMAL', 'NORMAL', '10']
pp_sample = preprocess(sample)
pred = forward(pp_sample, dt)


print(pred)