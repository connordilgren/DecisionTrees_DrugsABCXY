import csv
import numpy as np


def read_csv(csv_file):
    # import data -- get examples, attributes
    with open(csv_file, 'r') as f_csv:
        csv_reader = csv.reader(f_csv)

        # get attributes
        attributes = next(csv_reader)

        # get examples
        examples = []
        for example in csv_reader:
            examples.append(example)
    return examples, attributes


def get_attr_val_dict():
    # maps attr to list of possible values
    attr_val_dict = {
        'Age': [
            0,  # < 24
            1,  # < 34
            2,  # < 44
            3,  # < 54
            4,  # < 64
            5,  # >= 65
        ],
        'Sex': [
            'M',  # default
            'F'
        ],
        'BP': [
            'LOW',
            'NORMAL',  # default
            'HIGH'
        ],
        'Cholesterol': [
            'HIGH',
            'NORMAL'  # default
        ],
        'Na_to_K': [
            0,  # < 10
            1,  # < 15
            2,  # < 20
            3,  # < 30
            4,  # < 35
            5,  # >= 35
        ]}
    return attr_val_dict


def preprocess(original_samples):
    pp_samples = [x.copy() for x in original_samples]

    # sort Age into buckets: min_age onwards in 10 year increments
    age_i = attributes.index('Age')
    for sample in pp_samples:
        sample_age = int(sample[age_i])
        if sample_age < 24:
            sample[age_i] = 0
        elif sample_age < 34:
            sample[age_i] = 1
        elif sample_age < 44:
            sample[age_i] = 2
        elif sample_age < 54:
            sample[age_i] = 3
        elif sample_age < 64:
            sample[age_i] = 4
        else:
            sample[age_i] = 5

    # sort Na_to_K to buckets -- TODO: maximize information gained from splits
    na_i = attributes.index('Na_to_K')
    for sample in pp_samples:
        sample_na = float(sample[na_i])
        if sample_na < 10:
            sample[na_i] = 0
        elif sample_na < 15:
            sample[na_i] = 1
        elif sample_na < 20:
            sample[na_i] = 2
        elif sample_na < 25:
            sample[na_i] = 3
        elif sample_na < 35:
            sample[na_i] = 4
        else:
            sample[na_i] = 5

    return pp_samples


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


def decision_tree_learning(examples, attributes, parent_node, attr_val_dict):
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

        vk_s = attr_val_dict[A]
        for vk in vk_s:
            exs = [ex[:A_i] + ex[A_i+1:] for ex in examples if ex[A_i] == vk]
            subtree = decision_tree_learning(exs, remaining_attrs, tree, attr_val_dict)
            tree.add_child(subtree, vk)

    return tree


def forward(pp_sample, dt):
    # base case: dt is a classification
    if dt.is_leaf is True:
        return dt.pred

    # otherwise, recurse down tree
    test_attr = dt.test_attr
    test_attr_i = attributes.index(test_attr)
    samp_attr_v = pp_sample[test_attr_i]
    sub_dt = dt.children_nodes[samp_attr_v]
    return forward(pp_sample, sub_dt)


def split_data(pp_examples, train_split):
    # get num examples in train, test
    num_train = int(train_split * len(pp_examples))

    # random shuffle
    shuffled_data = pp_examples.copy()
    np.random.shuffle(shuffled_data)

    # split
    train_data = shuffled_data[:num_train]
    test_data = shuffled_data[num_train:]

    return train_data, test_data


def test_dt(dt, test_data):
    num_correct = 0
    for sample in test_data:
        pred = forward(sample, dt)
        if pred == sample[-1]:
            num_correct += 1
    acc = num_correct / len(test_data)
    return acc


def get_avg_acc(pp_examples, train_split, attributes, num_trials, attr_val_dict):
    accs = []

    for _ in range(num_trials):
        # split into train, test
        train_data, test_data = split_data(pp_examples, 0.8)

        # create decision tree
        dt = decision_tree_learning(train_data, attributes, None, attr_val_dict)

        # get accuracy on test_data
        acc = test_dt(dt, test_data)
        accs.append(acc)

    avg_acc = sum(accs) / num_trials
    return avg_acc


if __name__ == "__main__":

    # maps attr to list of possible values
    attr_val_dict = get_attr_val_dict()

    # get examples
    examples, attributes = read_csv("drug200.csv")

    # preprocess
    pp_examples = preprocess(examples)

    # get average performance for a given split
    avg_acc = get_avg_acc(pp_examples, 0.8, attributes, 20, attr_val_dict)

    print(avg_acc)
