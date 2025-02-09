    def visualize_tree(self):
        G = nx.DiGraph()
        labels = {}

        def add_node_edges(node: BlockChainNode):
            G.add_node(node.block.block_id, height=node.height)  # Add the height attribute here
            labels[node.block.block_id] = f"Miner {node.miner_id}\n Num Txs : {len(node.block.transactions)}"
            if node.parent:
                G.add_edge(node.parent.block.block_id, node.block.block_id)

            for child in node.children:
                add_node_edges(child)

        add_node_edges(self.blockchain_tree.root)

        pos = nx.multipartite_layout(G, subset_key="height", align='vertical')

        # Draw the graph with square-shaped nodes
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color="skyblue", font_size=5, font_weight="bold", width=2, edge_color="gray", node_shape='s')
        
        plt.title(f"Blockchain Tree of Peer {self.peer_id}")
        plt.show()