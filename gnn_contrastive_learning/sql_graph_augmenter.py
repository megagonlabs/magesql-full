import random
import json
from copy import deepcopy
import logging

import numpy as np
from nltk.corpus import words
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SQLGraphAugmenter:
    def __init__(self, src_schema_file, aug_schema_file, operator_probabilities=None, model=None, mask_token='<mask>'):
        """
        Initialize the SQL Graph Augmenter.
        :param schema_file: Path to the table schema JSON file.
        :param operator_probabilities: Dictionary of operator names and their probabilities.
        """
        # Load schema
        self.src_schema = self.load_schema_file(src_schema_file)
        logger.info(f"Loaded schema for {len(self.src_schema)} databases on the source split.")
        self.aug_schema = self.load_schema_file(aug_schema_file)
        logger.info(f"Loaded schema for {len(self.aug_schema)} databases on the augment split.")

        # Set default probabilities if none are provided
        if operator_probabilities is None:
            self.operator_probabilities = {
                "feature_masking": 0.7,
                "value_replacement": 0.3,
                "keyword_replacement": 0.3,
                "predicate_modification": 0.3,
                "join_node_removal": 0.2,
                "table_column_replacement": 0.2,
            }
        else:
            self.operator_probabilities = operator_probabilities

        self.operator_configs = {
            "feature_masking": {"graph": None},
            "value_replacement": {"graph": None},
            "keyword_replacement": {"graph": None},
            "predicate_modification": {"graph": None},
            "join_node_removal": {"graph": None},
            "table_column_replacement": {"graph": None, "db_id": None},
        }
        # Define augmenter configurations
        self.augmenter_configs = {
            "feature_masking": {
                "function": self.feature_masking,
                "kwargs": {"mask_prob": 0.5, "flag_inplace": True}
            },
            "value_replacement": {
                "function": self.value_replacement,
                "kwargs": {"db_id": "example_db", "val_replace_prob": 0.8, "flag_inplace": True}
            },
            "keyword_replacement": {
                "function": self.keyword_replacement,
                "kwargs": {"keyword_replace_prob": 0.7, "flag_inplace": True}
            },
            "predicate_modification": {
                "function": self.predicate_modification,
                "kwargs": {"drop_prob_where": 0.3, "simplify_conditions_prob": 0.5, "drop_having_prob": 0.2}
            },
            "join_node_removal": {
                "function": self.join_node_removal,
                "kwargs": {"drop_join_prob": 0.8, "flag_inplace": True}
            },
            "table_column_replacement": {
                "function": self.table_column_replacement,
                "kwargs": {"db_id": "", "replace_table_prob": 1.0, "flag_inplace": True}
            }
        }


        # Set mask token
        if model is not None:
            try:
                self.mask_token = model.tokenizer.mask_token
            except AttributeError:
                self.mask_token = mask_token
        else:
            self.mask_token = mask_token
        if not self.mask_token:
            self.mask_token = '<mask>'
        logger.info(f"Using mask token: {self.mask_token}")


    def load_schema_file(self, schema_file_path):
        """Load the table schema for the dataset
        """
        with open(schema_file_path, 'r') as f:
            schema = json.load(f)
        ## change list of tables to dict of tables
        schema = {table['db_id']: table for table in schema}
        
        for db_id, db_schema in schema.items():
            mapping = {}
            table_names = db_schema['table_names_original']
            for col_name, col_type in zip(db_schema['column_names_original'], db_schema['column_types']):
                table_idx, col_name = col_name
                if table_idx < 0:
                    continue
                table_name = table_names[table_idx]
                if table_name not in mapping:
                    mapping[table_name] = {}
                mapping[table_name][col_name] = col_type
            db_schema['col_type_mapping'] = mapping
        return schema
    
    def get_tables_and_columns(self, graph):
        """
        Extract all table nodes and their associated columns from the graph.
        :param graph: The SQL graph in NetworkX format.
        :return: Dictionary of tables and their columns.
        """
        tables_and_columns = {}
        for node, data in graph.nodes(data=True):
            if data["type"] == "TABLE":
                table_name = data["val"]
                columns = [
                    graph.nodes[col]["val"]
                    for col in graph.successors(node)
                    if graph.nodes[col]["type"] == "COLUMN"
                ]
                tables_and_columns[table_name] = columns
        return tables_and_columns

    def generate_positive_instance_with_args(self, graph, db_id=None, operator_probabilities=None, num_augmenters=1, augmenter_configs=None):
        """
        Generate a positive instance by applying multiple unique augmenters with specific arguments.

        :param graph: The original graph (NetworkX format).
        :param operator_probabilities: Dictionary of augmenter names and their probabilities.
        :param num_augmenters: Number of augmenters to sample.
        :param augmenter_configs: Dictionary mapping augmenter names to functions and arguments.
        :return: Augmented graph.
        """
        operator_probabilities = operator_probabilities or self.operator_probabilities
        augmenter_configs = augmenter_configs or self.augmenter_configs
        # Sample unique augmenters
        sampled_augmenters = self.sample_unique_augmenters(self.operator_probabilities, num_augmenters)
        logger.debug(f"Sampled augmenters: {sampled_augmenters}")

        # Create a deep copy of the graph
        augmented_graph = deepcopy(graph)

        # Apply the sampled augmenters sequentially
        for augmenter_name in sampled_augmenters:
            augmenter_config = augmenter_configs.get(augmenter_name)
            if augmenter_config is None:
                raise ValueError(f"Augmenter configuration for {augmenter_name} not found.")
            
            augmenter_function = augmenter_config["function"]
            ## handle db_id if it is required
            augmenter_kwargs = augmenter_config.get("kwargs", {}).copy()
            if "db_id" in augmenter_kwargs:
                augmenter_kwargs["db_id"] = db_id
            augmenter_args = augmenter_config.get("args", {})
            augmenter_kwargs = augmenter_config.get("kwargs", {})

            logger.debug(f"Applying augmenter: {augmenter_name} with args: {augmenter_args} and kwargs: {augmenter_kwargs}")
            # Apply the augmenter function with the specified arguments
            augmented_graph = augmenter_function(augmented_graph, *augmenter_args, **augmenter_kwargs)

        return augmented_graph

    def apply_random_augmentation(self, graph, db_id):
        """
        Apply a single augmentation operator based on probabilities.
        :param graph: The SQL graph to augment (in NetworkX format).
        :param db_id: Database ID for schema-aware operations.
        :return: Augmented graph.
        """
        # Select an operator based on probabilities
        operator = random.choices(
            list(self.operator_probabilities.keys()),
            weights=list(self.operator_probabilities.values()),
            k=1
        )[0]

        # Get the operator's default arguments
        operator_config = self.operator_configs.get(operator, {})
        operator_args = operator_config.copy()

        # Dynamically populate arguments that require runtime values
        if "graph" in operator_args:
            operator_args["graph"] = graph
        if "db_id" in operator_args:
            operator_args["db_id"] = db_id

        # Call the operator with the resolved arguments
        return getattr(self, operator)(**operator_args)


    def sample_unique_augmenters(self, operator_probabilities, num_augmenters=1):
        """
        Efficiently sample `num_augmenters` unique augmenters based on probabilities without replacement.

        :param operator_probabilities: Dictionary of augmenter names and their probabilities.
        :param num_augmenters: Number of augmenters to sample.
        :return: List of `num_augmenters` unique augmenter names.
        """
        if num_augmenters > len(operator_probabilities):
            raise ValueError("num_augmenters cannot exceed the number of available operators.")

        operators = np.array(list(operator_probabilities.keys()))
        probabilities = np.array(list(operator_probabilities.values()))

        # Normalize probabilities
        probabilities = probabilities / probabilities.sum()

        # Use numpy's choice for sampling without replacement
        sampled_augmenters = np.random.choice(operators, size=num_augmenters, replace=False, p=probabilities)

        return sampled_augmenters.tolist()

    def apply_multiple_augmentations(self, graph, db_id, num_augmentations=2):
        """
        Apply multiple augmentation operators sequentially.
        :param graph: The SQL graph to augment (in NetworkX format).
        :param db_id: Database ID for schema-aware operations.
        :param num_augmentations: Number of augmentations to apply.
        :return: Augmented graph.
        """
        augmented_graph = graph
        for _ in range(num_augmentations):
            augmented_graph = self.apply_random_augmentation(augmented_graph, db_id)
        return augmented_graph


    def feature_masking(self, graph, mask_prob:float=0.25, flag_inplace=True):
        """
        Node feature masking operator.
        :param graph: The SQL graph to augment in networkx format.
        :param db_id: Database ID (not used here).
        :return: Augmented graph.
        """
        if not flag_inplace:
            graph = deepcopy(graph)
        critical_keywords = {"SELECT", "JOIN", "WHERE", "GROUP", "ORDER"}
        for node, data in graph.nodes(data=True):
            # skip node that belongs to critical keywords
            if data['type'] == 'ROOT' or (data["type"] == "SQL_KEYWORD" and data.get("val", "").upper()  in critical_keywords):
                continue
            if np.random.rand() < mask_prob:
                # Keep the label intact but modify the node text to include [MASK]
                masked_text = f"{self.mask_token}"
                data["aug_text"] = masked_text
                logger.debug(f"Masking node '{node}' with text '{masked_text}'.")
        return graph


    def value_replacement(self, graph, db_id=None, val_replace_prob: float = 0.5, flag_inplace=True):
        """
        Value replacement operator with contextual and pattern-aware replacements.
        :param graph: The SQL graph to augment in NetworkX format.
        :param db_id: Database ID for schema-aware operations.
        :param val_replace_prob: Probability of replacing a value.
        :param flag_inplace: If False, creates a deepcopy of the graph before modifying.
        :return: Augmented graph.
        """
        if not flag_inplace:
            graph = deepcopy(graph)

        # Load nltk word corpus
        word_list = words.words()

        # Extract VALUE nodes
        value_nodes = [node for node, data in graph.nodes(data=True) if data["type"] == "VALUE"]
        if not value_nodes:
            logger.debug(f"No VALUE nodes found in the graph for db_id {db_id}. Skipping Value Replacement.")
            return graph

        for node in value_nodes:
            if graph.nodes[node].get("aug_text"):
                # Skip masked nodes
                continue
            if np.random.rand() > val_replace_prob:
                continue

            node_data = graph.nodes[node]
            current_value = node_data["val"]
            current_value = current_value.lstrip('VALUE_')

            # Infer the data type of the value
            if isinstance(current_value, str) and current_value.isdigit():
                current_value = int(current_value)  # Convert to integer for processing
            if isinstance(current_value, int):  # Handle integers
                num_digits = len(str(current_value))  # Determine number of digits
                lower_bound = 10 ** (num_digits - 1)
                upper_bound = (10 ** num_digits) - 1
                new_value = random.randint(lower_bound, upper_bound)
                if isinstance(node_data["val"], str):
                    new_value = str(new_value)  # Convert back to string if originally a string
            elif isinstance(current_value, float):  # Handle floats
                if 0.01 < current_value < 1.0:
                    new_value = round(random.uniform(0.01, 1.0), 2)
                elif current_value < 0.01:
                    new_value = round(random.uniform(0.0001, 0.01), 4)
                else:
                    new_value = round(random.uniform(1.0, 100.0), 2)
            elif isinstance(current_value, str):  # Handle strings
                if "%" in current_value:  # Handle wildcards in LIKE conditions
                    # Split into wildcard and value
                    parts = current_value.split("%")
                    new_value = "%".join(
                        random.choice(word_list) if part.strip() else "" for part in parts
                    )
                else:
                    # Replace with a random word
                    new_value = random.choice(word_list)
            elif isinstance(current_value, (bool, str)) and str(current_value).upper() in {"TRUE", "FALSE"}:  # Handle booleans
                new_value = "FALSE" if str(current_value).upper() == "TRUE" else "TRUE"
            elif isinstance(current_value, int) and current_value in {0, 1}:  # Handle boolean-like integers
                new_value = 1 if current_value == 0 else 0
            else:
                # Skip unsupported types
                logger.debug(f"Skipping node {node} with unsupported value type: {type(current_value)}")
                continue

            # Determine the identifier for the new value
            new_identifier = f"VALUE_{new_value}"

            # Check if the new value already exists in the graph
            if new_identifier in graph.nodes:
                logger.debug(f"Value '{new_value}' already exists. Merging nodes.")
                # Merge the current node with the existing node
                existing_node = new_identifier
                # Redirect all edges pointing to the current node to the existing node
                for src, _ in list(graph.in_edges(node)):
                    graph.add_edge(src, existing_node)
                for _, tgt in list(graph.out_edges(node)):
                    graph.add_edge(existing_node, tgt)
                # Remove the old node
                graph.remove_node(node)
            else:
                # Update the node with the new value and identifier
                logger.debug(f"Replacing value node '{node}' (current: {current_value}) with new value '{new_value}'.")
                graph.nodes[node].update({
                    "val": new_value,
                    "label": new_identifier
                })
                # nx.relabel_nodes(graph, {node: new_identifier}, copy=False)
                graph = nx.relabel_nodes(graph, {node: new_identifier}, copy=True)

        return graph


    def keyword_replacement(self, graph, keyword_replace_prob: float = 0.5, flag_inplace: bool = True):
        """
        Keyword replacement operator for toggling SQL keywords.
        :param graph: The SQL graph to augment in NetworkX format.
        :param keyword_replace_prob: Probability of replacing a keyword.
        :param flag_inplace: If False, creates a deepcopy of the graph before modifying.
        :return: Augmented graph.
        """
        if not flag_inplace:
            graph = deepcopy(graph)

        # Define replacement rules for SQL keywords
        replacement_map = {
            # Logical operators and comparisons
            "EQ": ["NEQ", "GT", "LT"],
            "NEQ": ["EQ", "GT", "LT"],
            "GT": ["LT", "GTE", "LTE"],
            "GTE": ["GT", "LTE", "LT"],
            "LT": ["GT", "LTE", "GTE"],
            "LTE": ["LT", "GTE", "GT"],
            "AND": ["OR"],
            "OR": ["AND"],

            # Arithmetic operators
            "SUB": ["ADD", "MUL", "DIV"],
            "ADD": ["SUB", "MUL", "DIV"],
            "DIV": ["MUL", "ADD", "SUB"],
            "MUL": ["DIV", "ADD", "SUB"],

            # Aggregation functions
            "COUNT": ["SUM", "AVG", "MIN", "MAX"],
            "SUM": ["COUNT", "AVG", "MIN", "MAX"],
            "AVG": ["COUNT", "SUM", "MIN", "MAX"],
            "MIN": ["COUNT", "SUM", "AVG", "MAX"],
            "MAX": ["COUNT", "SUM", "AVG", "MIN"]
        }

        # Locate all keyword nodes in the graph
        keyword_nodes = [
            node for node, data in graph.nodes(data=True)
            if data["type"] == "SQL_KEYWORD"
        ]

        for node in keyword_nodes:
            if np.random.rand() > keyword_replace_prob:
                continue

            node_data = graph.nodes[node]
            current_keyword = node_data["val"].upper()  # Use uppercase for consistency

            if current_keyword in {"IN", "LIKE"}:
                # Special handling for 'IN'/'NOT IN' and 'LIKE'/'NOT LIKE'
                graph = self._toggle_not_prefix(graph, node, current_keyword)
            elif current_keyword in replacement_map:
                # Replace with a random alternative
                replacements = replacement_map[current_keyword]
                new_keyword = random.choice(replacements)
                logger.debug(f"Replacing keyword '{current_keyword}' with '{new_keyword}'.")

                # Compute the next available index for the new keyword
                next_index = self._get_next_keyword_index(graph, new_keyword)

                # Update node attributes
                new_label = f"SQL_KEYWORD_{new_keyword}_{next_index}"
                graph.nodes[node].update({
                    "val": new_keyword,
                    "label": new_label,
                    "type": "SQL_KEYWORD"
                })

                # Update node identifier in the graph
                # nx.relabel_nodes(graph, {node: new_label}, copy=False)
                graph = nx.relabel_nodes(graph, {node: new_label}, copy=True)
                
            else:
                logger.debug(f"Skipping keyword '{current_keyword}' (no replacements defined).")

        return graph


    def _toggle_not_prefix(self, graph, keyword_node, current_keyword):
        """
        Toggle between 'NOT IN'/'IN' or 'NOT LIKE'/'LIKE' by adding/removing 'NOT' nodes.
        Handles cases where 'NOT' is a separate node in the graph.
        :param graph: The SQL graph in NetworkX format.
        :param keyword_node: Node representing the 'IN' or 'LIKE' keyword.
        :param current_keyword: The current keyword ('IN' or 'LIKE').
        :return: Updated graph.
        """
        # Check for parent 'NOT' nodes
        parent_nodes = list(graph.predecessors(keyword_node))
        not_nodes = [
            parent for parent in parent_nodes
            if graph.nodes[parent]["type"] == "SQL_KEYWORD" and graph.nodes[parent]["val"].upper() == "NOT"
        ]

        if not_nodes:
            # Case 1: Remove the 'NOT' prefix
            for not_node in not_nodes:
                logger.debug(f"Removing 'NOT' node: {not_node}")
                # Redirect edges from 'NOT' to 'keyword_node'
                for src, _ in graph.in_edges(not_node):
                    graph.add_edge(src, keyword_node)
                # Remove the 'NOT' node
                graph.remove_node(not_node)
        else:
            # Case 2: Add a 'NOT' prefix
            new_not_node = f"SQL_KEYWORD_NOT_{self._get_next_keyword_index(graph, 'NOT')}"
            logger.debug(f"Adding 'NOT' node: {new_not_node}")
            graph.add_node(new_not_node, label=new_not_node, type="SQL_KEYWORD", val="NOT")
            # Redirect edges to the 'NOT' node
            for src, _ in list(graph.in_edges(keyword_node)):
                graph.remove_edge(src, keyword_node)
                graph.add_edge(src, new_not_node)
            # Connect 'NOT' node to the keyword node
            graph.add_edge(new_not_node, keyword_node)

        return graph


    def _get_next_keyword_index(self, graph, keyword):
        """
        Get the next available index for a SQL keyword in the graph.
        :param graph: The SQL graph in NetworkX format.
        :param keyword: The SQL keyword (e.g., 'NOT').
        :return: Next available index as an integer.
        """
        keyword_nodes = [node for node, data in graph.nodes(data=True) if data["type"] == "SQL_KEYWORD" and data["val"].upper() == keyword.upper()]
        indices = [int(node.split("_")[-1]) for node in keyword_nodes if node.split("_")[-1].isdigit()]
        return max(indices, default=-1) + 1


    def safe_delete_node(self, graph, target_node):
        """
        Safely delete a node and its children from the graph.
        :param graph: The SQL graph in NetworkX format.
        :param target_node: The node to delete.
        """
        if not graph.has_node(target_node):
            logger.debug(f"Node '{target_node}' not found in the graph.")
            return graph
        ## remove the edges from parent_target_node to target_node
        parent_target_nodes = list(graph.predecessors(target_node))
        for parent_target_node in parent_target_nodes:
            graph.remove_edge(parent_target_node, target_node)

        # Collect all reachable nodes
        branch_nodes_set = set(nx.dfs_preorder_nodes(graph, target_node))
        branch_nodes_set.add(target_node)

        # Check if nodes in branch_children_set are referenced elsewhere
        unsafe_nodes = set()
        stack = [target_node]
        visited = set()

        while stack:
            current_node = stack.pop()
            if current_node in visited or current_node in unsafe_nodes:
                continue
            visited.add(current_node)

            is_unsafe = any(src not in branch_nodes_set for src, _ in graph.in_edges(current_node))
            if is_unsafe:
                unsafe_nodes.add(current_node)
                ## stop the traversal because the node and all children are unsafe
            else:
                # Add successors to stack if current node is safe
                stack.extend(graph.successors(current_node))

        # Nodes that are safe to delete
        safe_to_delete = branch_nodes_set - unsafe_nodes

        # Remove nodes, networkx will automatically remove edges
        graph.remove_nodes_from(safe_to_delete)
        
        logger.debug(f"Deleted target node: {target_node}")   
        logger.debug(f"Deleted nodes under target node: {safe_to_delete}")
        return graph


    def simplify_conditions_in_where(self, graph, where_node):
        """
        Simplify conditions in the WHERE clause by removing one condition (AND/OR node).
        :param graph: The SQL graph in NetworkX format.
        :param where_node: The WHERE node in the graph.
        :return: Modified graph.
        """
        # Collect all AND/OR nodes under the WHERE clause
        and_or_nodes = [
            node for node in nx.dfs_preorder_nodes(graph, source=where_node)
            if graph.nodes[node]["type"] == "SQL_KEYWORD" and graph.nodes[node]["val"].upper() in {"AND", "OR"}
        ]

        if not and_or_nodes:
            logger.debug(f"No AND/OR nodes found under WHERE clause {where_node}.")
            return graph

        # Randomly select an AND/OR node to modify
        target_node = random.choice(and_or_nodes)
        parent_nodes = list(graph.predecessors(target_node))

        if not parent_nodes:
            logger.debug(f"No parent found for node {target_node}. Skipping.")
            return graph
        if len(parent_nodes) > 1:
            logger.debug(f"Unexpected number of parents for node {target_node}. Skipping.")
            return graph
        
        parent_of_target_node = parent_nodes[0]

        # Get all one hop children of the selected AND/OR node
        children = list(graph.successors(target_node))
        if len(children) != 2:
            logger.debug(f"Unexpected number of children for node {target_node}, there is {len(children)} children. Skipping.")
            return graph
        child_and_or = [child for child in children if graph.nodes[child]["val"].upper() in {"AND", "OR"}]

        if child_and_or:
            # Case 1: target node has a child that is another AND/OR
            ## randomly select one child to keep
            child_keep = random.choice(child_and_or)
            graph.add_edge(parent_of_target_node, child_keep)  # Redirect edges from parent_of_target_node to child_keep
            graph.remove_edge(parent_of_target_node, target_node)  # Remove edge from parent_of_target_node to target_node
            graph.remove_edge(target_node, child_keep)

            # Safely delete target_node and its other children
            logger.debug(f"Redirecting parent {parent_of_target_node} to child {child_keep} and deleting node {target_node}.")
            graph = self.safe_delete_node(graph, target_node)
        else:
            # Case 2: target node has two children that are expressions
            child_B1, child_B2 = children
            # Randomly decide which child to keep and which to drop
            drop_child, keep_child = random.choice([(child_B1, child_B2), (child_B2, child_B1)])

            # Redirect edges from parent_of_target_node to keep_child
            graph.add_edge(parent_of_target_node, keep_child)
            graph.remove_edge(parent_of_target_node, target_node)  # Remove edge from parent_of_target_node to target_node
            graph.remove_edge(target_node, keep_child)

            # Safely delete the drop_child and target_node
            logger.debug(f"Redirecting parent {parent_of_target_node} to child {keep_child} and deleting node {target_node}.")
            graph = self.safe_delete_node(graph, target_node)
        return graph


    def predicate_modification(self, graph, drop_prob_where=0.3, simplify_conditions_prob=0.5, drop_having_prob=0.2, flag_inplace=True):
        """
        Modify or drop predicates from the SQL graph.
        :param graph: The SQL graph in NetworkX format.
        :param drop_prob_where: Probability of dropping the WHERE clause.
        :param simplify_conditions_prob: Probability of simplifying compound WHERE conditions (AND/OR).
        :param drop_having_prob: Probability of dropping the HAVING clause.
        :param flag_inplace: If False, the function will create and return a deepcopy of the graph.
        :return: Modified graph.
        """
        if not flag_inplace:
            graph = deepcopy(graph)

        # Case 1: Drop WHERE Clause
        if np.random.rand() < drop_prob_where:
            where_nodes = [
                node for node, data in graph.nodes(data=True)
                if data["type"] == "SQL_KEYWORD" and data["val"].upper() == "WHERE"
            ]
            ## random drop one where clause
            if where_nodes:
                where_node = random.choice(where_nodes)
                logger.debug(f"Dropping WHERE clause: {where_node}")
                graph = self.safe_delete_node(graph, where_node)

        # Case 2: Simplify Conditions in WHERE Clause (AND/OR)
        if np.random.rand() < simplify_conditions_prob:
            where_nodes = [
                node for node, data in graph.nodes(data=True)
                if data["type"] == "SQL_KEYWORD" and data["val"].upper() == "WHERE"
            ]
            ## random simplify one where clause
            if where_nodes:
                where_node = random.choice(where_nodes)
                logger.debug(f"Simplifying conditions in WHERE clause: {where_node}")
                graph = self.simplify_conditions_in_where(graph, where_node)

        # Case 3: Drop HAVING Clause
        if np.random.rand() < drop_having_prob:
            having_nodes = [
                node for node, data in graph.nodes(data=True)
                if data["type"] == "SQL_KEYWORD" and data["val"].upper() == "HAVING"
            ]
            ## random drop one having clause
            if having_nodes:
                having_node = random.choice(having_nodes)
                logger.debug(f"Dropping HAVING clause: {having_node}")
                graph = self.safe_delete_node(graph, having_node)

        return graph

    
    def join_node_removal(self, graph, drop_join_prob=0.7, flag_inplace=True):
        """
        Remove one random join from a SELECT node if more than 2 joins exist.
        :param graph: The SQL graph in NetworkX format.
        :param drop_join_prob: Probability of applying the join removal operator.
        :return: Modified graph.
        """
        if not flag_inplace:
            graph = deepcopy(graph)

        # Identify SELECT nodes
        select_nodes = [
            node for node, data in graph.nodes(data=True)
            if data["type"] == "SQL_KEYWORD" and data["val"].upper() == "SELECT"
        ]

        for select_node in select_nodes:
            # Find all JOIN nodes connected to the SELECT node
            join_nodes = [
                child for child in graph.successors(select_node)
                if graph.nodes[child]["type"] == "SQL_KEYWORD" and graph.nodes[child]["val"].upper() == "JOIN"
            ]

            # If there are >= 2 joins, randomly drop one
            if len(join_nodes) > 1:
                if np.random.rand() > drop_join_prob:
                    continue
                join_to_remove = random.choice(join_nodes)
                logger.debug(f"Removing join node: {join_to_remove}")

                # Safely delete the join node and its subtree
                graph.remove_edge(select_node, join_to_remove)
                graph = self.safe_delete_node(graph, join_to_remove)

        return graph


    def table_column_replacement(self, graph, db_id, replace_table_prob=1.0, flag_inplace=True):
        """
        Replace tables and columns in the SQL graph with those from a randomly selected database in aug_schema.
        :param graph: The SQL graph in NetworkX format.
        :param db_id: The source database ID in src_schema.
        :param replace_table_prob: Probability of applying table-column replacement.
        :return: Modified graph.
        """
        if not flag_inplace:
            graph = deepcopy(graph)
        if np.random.rand() > replace_table_prob:
            return graph  # Skip augmentation based on probability

        # Step 1: Identify all table and column nodes in the graph
        table_ids = [
            node for node, data in graph.nodes(data=True)
            if data["type"] == "TABLE"
        ]
        col_ids = [
            node for node, data in graph.nodes(data=True)
            if data["type"] == "COLUMN"
        ]

        if not table_ids:
            logger.debug(f"No table nodes found in the graph for db_id {db_id}. Skipping.")
            return graph

        num_tables = len(table_ids)

        # Step 2: Filter suitable augmentation schemas
        candidate_dbs = [
            aug_db_id for aug_db_id, aug_schema in self.aug_schema.items()
            if len(aug_schema["col_type_mapping"]) >= num_tables
        ]
        if not candidate_dbs:
            logger.debug(f"No suitable augmentation databases found for db_id {db_id}. Skipping.")
            return graph

        # Randomly select an augmentation schema
        selected_aug_db_id = random.choice(candidate_dbs)
        aug_schema = self.aug_schema[selected_aug_db_id]
        logger.debug(f"Selected augmentation database: {selected_aug_db_id}")

        # Step 3: Map original tables and columns to augmented schema
        table_id_mapping = {}  # Map original table IDs to augmented table IDs
        table_attrs_mapping = {}  # Map original table names to augmented table names and IDs

        col_id_mapping = {}  # Map original column IDs to augmented column IDs
        col_attrs_mapping = {}  # Map original column names to augmented column names and IDs
        
        for ori_table_id in table_ids:
            ori_table_name = graph.nodes[ori_table_id]["val"]
            aug_tables_name = list(aug_schema["col_type_mapping"].keys())
            aug_table_name = random.choice(aug_tables_name)
            aug_table_id = f"TABLE_{aug_table_name}"
            table_id_mapping[ori_table_id] = aug_table_id
            table_attrs_mapping[aug_table_id] = {
                'label': aug_table_id,
                'val': aug_table_name,
                'type': 'TABLE'
            }

            # Replace columns
            ori_cols = [
                col for col in col_ids
                if col.startswith(f"TABLE_{ori_table_name}_COLUMN_")
            ]
            if not ori_cols:
                logger.debug(f"No columns found for table {ori_table_name}. Skipping.")
                continue

            aug_cols = [
                col for col, col_type in aug_schema["col_type_mapping"].get(aug_table_name, {}).items()
            ]
            if not aug_cols:
                logger.debug(f"No columns found for aug table {aug_table_name}. Skipping.")
                continue

            # Match columns by type if possible
            for ori_col_id in ori_cols:
                ori_col_name = graph.nodes[ori_col_id]["val"]
                try:
                    ori_col_type = self.src_schema[db_id]["col_type_mapping"][ori_table_name].get(ori_col_name)
                except KeyError:
                    logger.debug(f"Column type not found for {ori_col_name} in table {ori_table_name}. Skipping.")
                    ori_col_type = 'unknown'
                    continue
                # Find columns with matching types
                matching_aug_cols = [
                    aug_col for aug_col in aug_cols
                    if aug_schema["col_type_mapping"][aug_table_name].get(aug_col) == ori_col_type
                ]
                if matching_aug_cols:
                    selected_aug_col = random.choice(matching_aug_cols)
                else:
                    selected_aug_col = random.choice(aug_cols)  # Fallback to random
                selected_aug_col_id = f"TABLE_{aug_table_name}_COLUMN_{selected_aug_col}"
                col_id_mapping[ori_col_id] = selected_aug_col_id
                col_attrs_mapping[selected_aug_col_id] = {
                    'label': selected_aug_col_id,
                    'val': selected_aug_col,
                    'type': 'COLUMN'
                }

        # Step 4: Update the graph
        # nx.relabel_nodes(graph, table_id_mapping, copy=False)
        graph = nx.relabel_nodes(graph, table_id_mapping, copy=True)
        nx.set_node_attributes(graph, table_attrs_mapping)
        # nx.relabel_nodes(graph, col_id_mapping, copy=False)
        graph = nx.relabel_nodes(graph, col_id_mapping, copy=True)
        nx.set_node_attributes(graph, col_attrs_mapping)

        return graph
