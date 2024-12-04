from collections import Counter, defaultdict

from entscheidungsbaum import DecisionNode, DecisionTree, EndNode, Node


class DecisionTreeLearner:
    def __init__(
        self, data: list[dict], target_feature: str, ignored_features: list[str] = []
    ) -> None:
        self.ALL_DATA = data
        self.FEATURES = defaultdict(set)
        for row in self.ALL_DATA:
            for feature, value in row.items():
                self.FEATURES[feature].add(value)
        assert (
            target_feature in self.FEATURES
        ), "Das zu lernende Feature kommt in den Daten nicht vor!"
        self.TARGET_FEATURE = target_feature
        self.TARGET_VALUES = self.FEATURES[target_feature]
        self.IGNORED_FEATURES = ignored_features

    def split_data(self, data, feature):
        splits = defaultdict(list)
        for row in data:
            value = row[feature]
            splits[value].append(row)
        return splits

    def gini_impurity(self, data):
        feature = self.TARGET_FEATURE
        values = self.TARGET_VALUES
        total_num = len(data)
        cnt = Counter(row[feature] for row in data)
        rel_haeufigkeiten = [cnt[val] / total_num for val in values]
        return 1 - sum(p * p for p in rel_haeufigkeiten)

    def weighted_gini(self, data, feature):
        num_total = len(data)
        splits = self.split_data(data, feature)
        gini_total = 0
        for val in splits:
            rows = splits[val]
            num_rows = len(rows)
            gini = self.gini_impurity(rows)
            weight = num_rows / num_total
            gini_total += weight * gini
        return gini_total

    def best_feature(self, data, possible_features):
        best_gini = 1  # Worst case
        best_f = None
        for feature in possible_features:
            new_gini = self.weighted_gini(data, feature)
            if new_gini < best_gini:
                best_gini = new_gini
                best_f = feature
        return best_f, best_gini

    def learn(self, depth):
        dt = DecisionTree()
        data = list(self.ALL_DATA)  # Kopie
        possible_features = (
            set(self.FEATURES.keys())
            - {self.TARGET_FEATURE}
            - set(self.IGNORED_FEATURES)
        )
        dt.root = self.learn_recursively(data, possible_features, depth)
        return dt

    def learn_recursively(self, data, possible_features, depth) -> Node:
        """Die entscheidende Funktion, die den Baum rekursiv aufbaut."""
        data_target_values = [row[self.TARGET_FEATURE] for row in data]
        counter = Counter(data_target_values)
        current_gini = self.gini_impurity(data)
        if depth == 0 or current_gini == 0:
            # Basisfall: Endknoten erzeugen
            most_common = counter.most_common(1)[0][0]  # häufigstes Element bestimmen
            new_node = EndNode(self.TARGET_FEATURE, most_common)
        else:
            # Rekursiver Fall: Feature mit geringstem gewichteten Gini-Koeffizienten bestimmen
            best_feature, best_gini = self.best_feature(data, possible_features)
            # Anhand der verschiedenen Werte des Features die Daten aufteilen
            splits = self.split_data(data, best_feature)
            new_node = DecisionNode(
                best_feature, splits.keys()
            )  # neuer Entscheidungsknoten
            remaining_feautures = possible_features - {best_feature}
            for value, split_data in splits.items():
                # Rekursiver Aufruf für jeden Teilbaum
                new_node.edges[value] = self.learn_recursively(
                    split_data, remaining_feautures, depth - 1
                )
        new_node.info["gini"] = current_gini
        new_node.info["counter"] = counter
        return new_node
