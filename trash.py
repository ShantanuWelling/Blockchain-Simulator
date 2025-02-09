# Find the longest chain by recursively checking the depth of the tree
    node_colors = []
    def find_longest_chain(node: BlockChainNode):
        # Start from the node and go to its children to find the longest chain
        longest_chain = [node]  # The current node is part of the chain
        max_depth = 0

        # Find the longest chain among children
        for child in node.children:
            child_chain = find_longest_chain(child)
            if len(child_chain) > max_depth:
                longest_chain = child_chain
                max_depth = len(child_chain)

        return longest_chain

    # Find the longest chain starting from the root
    longest_chain = find_longest_chain(self.blockchain_tree.root)

    # Mark nodes in the longest chain as green
    for node in longest_chain:
        longest_chain_nodes.add(node.block.block_id)
    for node in G.nodes:
      if node in longest_chain_nodes:
          node_colors.append('green')
      else:
          node_colors.append('blue')
