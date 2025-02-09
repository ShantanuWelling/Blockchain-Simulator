def visualize_tree(self):
        G = nx.DiGraph()
        labels = {}

        def add_node_edges(node: BlockChainNode):
            G.add_node(node.block.block_id, height=node.height)  # Add the height attribute here
            labels[node.block.block_id] = f"Miner {node.miner_id}\n N-Txs : {len(node.block.transactions)}\n Mine time: {node.block.create_timestamp}\n Receive time: {node.receive_timestamp}"
            if node.parent:
                G.add_edge(node.parent.block.block_id, node.block.block_id)

            for child in node.children:
                add_node_edges(child)

        add_node_edges(self.blockchain_tree.root)

        pos = nx.multipartite_layout(G, subset_key="height", align='horizontal')

        # Invert the height to ensure the root node is at the top
        min_height = min([node[1]['height'] for node in G.nodes(data=True)])
        max_height = max([node[1]['height'] for node in G.nodes(data=True)])
        
        for node, (x, y) in pos.items():
            # Invert the y-axis to place root at the top
            pos[node] = (x, max_height - y + min_height)

        # Draw the graph with square-shaped nodes
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color="skyblue", font_size=5, font_weight="bold", width=2, edge_color="gray", node_shape='s')
        
        plt.title(f"Blockchain Tree of Peer {self.peer_id}")
        plt.show()