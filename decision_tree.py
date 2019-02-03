import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None
    def counter(self,array):
        a = np.array(array)
        unique, counts = np.unique(a, return_counts=True)
        return dict(zip(unique, counts))

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        #      A B C D E F cls
        # x1  [0 0 2 2 0 0][1]
        # x2  [0 0 2 2 0 1][0]
        self.root_node = TreeNode(features,labels,len(np.unique(labels)))
        self.root_node.feature_uniq_split = set([i for i in range(len(features[0]))])
        
        queue = [self.root_node]
        while len(queue) != 0:
            current_node = queue.pop(0)
            
            if len(np.unique(current_node.labels)) < 2 or len(current_node.feature_uniq_split) < 1:
                current_node.splittable = False
            else:
                
                current_node.splittable = True

                class_val_ordered = np.unique(current_node.labels)
                class_unique_type,count_class = np.unique(current_node.labels,return_counts=True)
                current_node_entropy = Util.entropy(count_class)
                # for each current node check 
                max_IG = float('-inf')
                max_atr = None
                max_attri_val = None
                for attri_index,data in enumerate(np.array(current_node.features).T):
                    if attri_index not in current_node.feature_uniq_split:
                        continue                                            
                    attribute_table = np.zeros((len(np.unique(data)),current_node.num_cls)) # for each attribute count each value and corresponding class
                    attri_val_ordered = np.unique(data) # possible attri val                
                    attri_class_index = {c: (data==c).nonzero()[0] for c in np.unique(data)} # get the index array for the class
            
                    for each_type_index in range(len(attri_val_ordered)):                    
                        count_list = np.take(current_node.labels,attri_class_index[attri_val_ordered[each_type_index]])
                        for selected_class_type,value in self.counter(count_list).items():
                            attribute_table[each_type_index][np.where(class_val_ordered==selected_class_type)] = value
                    IG = Util.Information_Gain(current_node_entropy,attribute_table)

                    if max_IG < IG:
                        max_IG = IG
                        max_atr = attri_index
                        current_node.attri_val_list = attri_val_ordered

                    elif max_IG == IG:
                        if len(current_node.attri_val_list) < len(attri_val_ordered):
                            max_IG = IG
                            max_atr = attri_index
                            current_node.attri_val_list = attri_val_ordered
                        elif len(current_node.attri_val_list) == len(attri_val_ordered):
                            if  max_atr > attri_index:
                                max_IG = IG
                                max_atr = attri_index
                                current_node.attri_val_list = attri_val_ordered                       
                        else:
                            continue

                    else:
                        continue

                current_node.dim_split = max_atr
                # complete the remain attribute 
                current_node.split()
                for each_child in current_node.children:
                    queue.append(each_child)
        # Util.print_tree(self)
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features 
        self.labels = labels
        self.children = [] 
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split
        self.attri_val_list = None # current node's possible attribute list 
        self.feature_uniq_split = None  # the possible unique values of the feature to be split


    def split(self):
        if self.splittable == True:            
            # print (self.attri_val_list)
            for each_val in sorted(self.attri_val_list): # attribute possible val
                splitted_data = []
                splitted_labels = []
                for i in range(len(self.features)):
                    if self.features[i][self.dim_split] == each_val:
                        splitted_data.append(self.features[i])
                        splitted_labels.append(self.labels[i])
                child = TreeNode(splitted_data,splitted_labels,len(np.unique(splitted_labels)))
                self.feature_uniq_split.remove(self.dim_split)
                # print (self.feature_uniq_split)
                child.feature_uniq_split = set(self.feature_uniq_split)
                self.children.append(child)
                self.feature_uniq_split.add(self.dim_split)
        else:
            return

    def predict(self, feature):
        # feature: List[any]
        # return: int
        node = self
        while node.splittable != False or len(node.children) >= 1:
            # print (feature,node.dim_split)
            if feature[node.dim_split] not in set(node.attri_val_list):
                return node.cls_max
            for index, each_val in enumerate(node.attri_val_list):
                if each_val == feature[node.dim_split]:
                    node = node.children[index]
                    break
        return node.cls_max
