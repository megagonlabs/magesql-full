import os
import sys
import uuid
from collections import defaultdict
import re

import sqlglot
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

def print_edges_by_nodes(G):
    for node in G.nodes():
        print(f"Node: {node}")
        for edge in G.edges(node):
            print(f"Edge: {edge}")

def plot_graph(G, flag_save=False):
    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'label')
    node_colors = {
        "SQL_KEYWORD": "orange",
        "TABLE": "lightblue",
        "COLUMN": "lightcoral",
        "VALUE": "lightgreen",
        "ROOT": "gray",
    }
    color_map = [node_colors[G.nodes[node]["type"]] for node in G]

    plt.figure(figsize=(15, 10))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
            node_color=color_map, node_size=2000, font_size=10, font_weight="bold", edge_color="black")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="gray")
    plt.title("SQL Query AST Graph")
    plt.pause(1)
    plt.show()
    if flag_save:
        ## save the plot
        plt.savefig("sql_query_ast.png")


def encode_texts_in_batches(texts, model, tokenizer, batch_size=128):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings.extend(batch_embeddings.cpu().numpy())
    return embeddings

def replace_double_quotes(sql):
    sql = re.sub(r'"(\w+)"', lambda match: f"'{match.group(1)}'", sql)
    return sql

class SQL2Graph:
    def __init__(self):
        self.keyword_count = defaultdict(int)
        self.node_to_expression = {}
        self.select_stack = []
        self.alias_mapping = {}

    def reset(self):
        self.keyword_count = defaultdict(int)
        self.node_to_expression = {}
        self.select_stack = []
        self.alias_mapping = {}
    
    def add_unique_node(self, graph, identifier, value, entity_type, expression=None):
        if identifier not in graph:
            graph.add_node(identifier, label=identifier, val=value, type=entity_type)
            if expression:
                self.node_to_expression[identifier] = expression  # Map to the expression
        return identifier

    def add_unique_edge(self, graph, source, target, edge_label=None):
        if not graph.has_edge(source, target):
            graph.add_edge(source, target, label=edge_label)

    def create_alias_mapping(self, node, graph):
        """
        First traversal: Create alias mappings and add table nodes to the graph.
        """
        if isinstance(node, sqlglot.expressions.Table):
            table_name = node.this.name
            alias = node.args.get("alias")
            if alias:
                self.alias_mapping[alias.name] = table_name
            # Add the table node to the graph
            table_label = f"TABLE_{table_name}"
            self.add_unique_node(graph, table_label, table_name, "TABLE")
        # Recursively process children
        for child in node.args.values():
            if isinstance(child, list):
                for sub_child in child:
                    self.create_alias_mapping(sub_child, graph)
            elif isinstance(child, sqlglot.expressions.Expression):
                self.create_alias_mapping(child, graph)

    def process_column_node(self, column_node, parent_node_label, graph):
        # Look up the parent Select node's expression from the mapping
        parent_expression = self.node_to_expression.get(parent_node_label)
        if not parent_expression:
            raise ValueError(f"Parent node {parent_node_label} not found in mapping.")
        
        # Get the current SELECT context from the stack
        if not self.select_stack:
            raise ValueError("SELECT stack is empty; cannot resolve column context.")
        current_select = self.select_stack[-1]

        # Check if column has an explicit table
        from_clause = None
        if column_node.table:
            if isinstance(column_node.table, str):
                table_name = self.alias_mapping.get(column_node.table, column_node.table)
            else:
                # Use the explicit table name (resolve aliases if needed)
                table_name = self.alias_mapping.get(column_node.table.name, column_node.table.name)
        else:
            # Infer table from the `FROM` clause of the current SELECT context
            from_clause = current_select.args.get("from")
            tables_in_scope = []  # List of tables in the `FROM` clause or `JOIN`s

            if from_clause and isinstance(from_clause.this, sqlglot.expressions.Table):
                # Single table in the FROM clause
                table_name = self.alias_mapping.get(from_clause.this.name, from_clause.this.name)
                tables_in_scope.append(table_name)
            elif current_select.args.get("joins"):
                # Handle multiple tables in JOINs
                for join in current_select.args["joins"]:
                    join_table = join.this
                    if isinstance(join_table, sqlglot.expressions.Table):
                        table_name = self.alias_mapping.get(join_table.this.name, join_table.this.name)
                        tables_in_scope.append(table_name)
            # If there's exactly one table in scope, use it
            if len(tables_in_scope) == 1:
                table_name = tables_in_scope[0]
            else:
                raise ValueError(
                    f"Ambiguous column '{column_node.this.name}': cannot resolve table. "
                    f"Tables in scope: {tables_in_scope}"
                )
        # Create or reuse nodes
        table_label = f"TABLE_{table_name}"
        column_label = f"{table_label}_COLUMN_{column_node.this.name}"
        self.add_unique_node(graph, column_label, column_node.this.name, "COLUMN")
        self.add_unique_node(graph, table_label, table_name, "TABLE")
        
        # Connect column to its table and parent SQL clause
        self.add_unique_edge(graph, column_label, table_label, edge_label="belongs_to")
        self.add_unique_edge(graph, parent_node_label, column_label, edge_label="expressions")


    def traverse_and_build_graph(self, node, graph, parent=None, edge_label=None):
        node_type = node.__class__.__name__.upper()

        # Process SQL keywords with unique labels
        if node_type in {"SELECT", "EXCEPT", "INTERSECT", "UNION", "JOIN", "WHERE", "GROUP", "ORDER", "LIMIT", "OFFSET", "HAVING", "LITERAL", "DISTINCT", "SUBQUERY", "SUB", "ADD", "DIV", "MUL", "PAREN", "NEG"}:
            label_type = node_type.upper()
            count = self.keyword_count[label_type]
            label = f"SQL_KEYWORD_{label_type}_{count}"
            self.add_unique_node(graph, label, label_type, "SQL_KEYWORD", expression=node)
            if parent:
                self.add_unique_edge(graph, parent, label, edge_label)

            parent = label
            self.keyword_count[label_type] += 1

            if node_type == "SELECT":
                # Push the current SELECT expression onto the stack
                self.select_stack.append(node)

            if node_type == "LITERAL":
                parent = label
                value = node.this
                value_label = f"VALUE_{value}"
                self.add_unique_node(graph, value_label, value, "VALUE", expression=node)
                self.add_unique_edge(graph, parent, value_label, edge_label="VALUE")
                return

        elif node_type in {"COUNT", "SUM", "AVG", "MIN", "MAX"}:
            func_label = f"SQL_KEYWORD_{node_type.upper()}_{self.keyword_count[node_type.upper()]}"
            self.add_unique_node(graph, func_label, node_type.upper(), "SQL_KEYWORD", expression=node)
            if parent:
                self.add_unique_edge(graph, parent, func_label, edge_label)
            self.keyword_count[node_type.upper()] += 1
            parent = func_label

        # Handle Star (*)
        elif node_type == "STAR":
            star_label = "SQL_KEYWORD_STAR"
            self.add_unique_node(graph, star_label, "*", "SQL_KEYWORD", expression=node)
            if parent:
                self.add_unique_edge(graph, parent, star_label, edge_label)

        # Handle Table expressions, using the table name directly from Identifier
        elif node_type == "TABLE":
            table_name = self.alias_mapping.get(node.this.name, node.this.name)
            label = f"TABLE_{table_name}"
            self.add_unique_node(graph, label, table_name, "TABLE", expression=node)
            if parent:
                self.add_unique_edge(graph, parent, label, edge_label)

        # Handle Column expressions
        elif node_type == "COLUMN":
            self.process_column_node(node, parent, graph)

        # Handle Conditional operators (AND, OR, NOT) and Comparisons
        elif node_type in {"AND", "OR", "NOT", "IN", "BETWEEN", "LIKE", "EQ", "NEQ", "GT", "GTE", "LT", "LTE"}:
            operator = node_type.upper()
            count = self.keyword_count[operator]
            operator_label = f"SQL_KEYWORD_{operator}_{count}"
            self.add_unique_node(graph, operator_label, operator, "SQL_KEYWORD", expression=node)
            if parent:
                self.add_unique_edge(graph, parent, operator_label, edge_label)
            self.keyword_count[operator] += 1
            parent = operator_label
        elif node_type == "IDENTIFIER":
            # Skip identifiers
            return
        elif node_type == "ORDERED":
            if isinstance(node.this, sqlglot.expressions.Column):
                self.process_column_node(node.this, parent, graph)
        else:
            print(f"Expression {node_type} is not supported in the currect graph building process")

        for key, child in node.args.items():
            # if isinstance(child, list) and not isinstance(child, str) and key != "joins":
            if isinstance(child, list) and not isinstance(child, str):
                for sub_child in child:
                    self.traverse_and_build_graph(sub_child, graph, parent, edge_label=key)

            if key == "this" and isinstance(child, sqlglot.expressions.Identifier):
                # Skip traversing identifiers directly
                # Skip alias as it's already processed in the first tranversal
                continue
            elif key == 'alias':
                # Skip traversing the alias directly
                continue
            elif key == "from":
                self.traverse_and_build_graph(child.this, graph, parent, edge_label="FROM")
            elif isinstance(child, sqlglot.expressions.Expression):
                self.traverse_and_build_graph(child, graph, parent, edge_label=key)
            else:
                # skip other types of args
                continue
        
        if node_type == "SELECT":
            self.select_stack.pop()
    
    def sql2networkx(self, sql:str, flag_replace_double_quotes=False):
        if flag_replace_double_quotes:
            sql = replace_double_quotes(sql)
        self.reset()
        expression = sqlglot.parse_one(sql)
        G = nx.DiGraph()
        root_node = self.add_unique_node(G, 'ROOT', 'ROOT', 'ROOT', expression=None)
        self.create_alias_mapping(expression, G)
        self.traverse_and_build_graph(expression, G, parent=root_node)
        return G

class SQL2GraphWithFeatures:
    """
    Convert SQL to graphs with features and PyG compatibility.
    """
    def __init__(self, batch_size=32, model_name='all-mpnet-base-v2'):
        self.sql_to_graph = SQL2Graph()
        self.model_name = model_name
        try:
            ## if the model is in the sentence-transformers
            self.model = SentenceTransformer(model_name)
        except ValueError:
            print(f"Model {model_name} not found. Using DistilBERT instead.")
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.batch_size = batch_size
        
        # Define node types for one-hot encoding
        self.node_types = ["ROOT", "SQL_KEYWORD", "TABLE", "COLUMN", "VALUE"]
        self.type_to_one_hot = {t: [1 if i == idx else 0 for i in range(len(self.node_types))]
                                for idx, t in enumerate(self.node_types)}
        
    def preprocess_nodes(self, G):
        node_texts = []
        for node, data in G.nodes(data=True):
            if "aug_text" in data: # used by augmentation
                identifier = data["aug_text"]
            else:
                identifier = data["label"]
            if identifier.startswith("SQL_KEYWORD"):
                ## remove the count number
                parts = identifier.split("_")
                if parts[-1].isdigit():
                    identifier = " ".join(parts[:-1])  # Remove suffix like "_0"
                else:
                    identifier = " ".join(parts) 
                node_texts.append(identifier)
            else:
                ## replace the _ with space
                node_texts.append(identifier.replace("_", " "))
        return node_texts

    def compute_node_features(self, G):
        node_texts = self.preprocess_nodes(G)
        ## if model is sentnece-transformers model
        if hasattr(self.model, "encode"):
            embeddings = self.model.encode(node_texts, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False).cpu().numpy()
        else:
            # Generate DistilBERT embeddings
            embeddings = []
            for i in range(0, len(node_texts), self.batch_size):
                batch_texts = node_texts[i:i + self.batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.extend(batch_embeddings.cpu().numpy())

        # Add one-hot encoding for node type
        node_features = {}
        for idx, (node, data) in enumerate(G.nodes(data=True)):
            node_type = data["type"]
            type_encoding = self.type_to_one_hot[node_type]
            embedding = embeddings[idx]
            node_features[node] = np.concatenate([type_encoding, embedding])
        return node_features

    def convert_to_pyg(self, G, bidirectional=True):
        """
        Convert a NetworkX graph to PyTorch Geometric Data format with an option for bidirectional edges.
        :param G: NetworkX graph to convert.
        :param bidirectional: Boolean flag to add reverse edges, making the graph bidirectional.
        :return: PyTorch Geometric Data object.
        """
        # Step 1: Map node identifiers to integer indices
        node_mapping = {node: idx for idx, node in enumerate(G.nodes)}

        # Step 2: Create edge index
        edge_index = torch.tensor(
            [[node_mapping[src], node_mapping[dst]] for src, dst in G.edges],
            dtype=torch.long
        ).t().contiguous()

        # If bidirectional, add reverse edges
        if bidirectional:
            reverse_edge_index = torch.flip(edge_index, [0])  # Reverse the edges
            edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)  # Concatenate

        # Step 3: Compute node features
        node_features_dict = self.compute_node_features(G)
        # Ensure features are in the same order as node indices
        node_features = [node_features_dict[node] for node in G.nodes]

        # Optimize: Convert the list of numpy arrays to a single numpy array before creating a tensor
        node_features = np.array(node_features)
        x = torch.tensor(node_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index)

    def sql_to_pyg(self, sql:str, flag_replace_double_quotes=False):
        nx_graph = self.sql_to_graph.sql2networkx(sql, flag_replace_double_quotes=flag_replace_double_quotes)
        pyg_graph = self.convert_to_pyg(nx_graph)
        return pyg_graph