import dendropy as dp

# construct a complete binary tree
taxa = ['taxon_' + str(i+1) for i in np.arange(2 ** 5)]
taxon_namespace = dp.TaxonNamespace(taxa)
tree = dp.Tree(taxon_namespace=taxon_namespace)
tree.taxon_namespace


# +
## create complete binary tree
def add_node(nd, tree):
    ch_left = dp.Node(edge_length=1)
    ch_right = dp.Node(edge_length=1)
    nd.set_child_nodes([ch_left, ch_right])

add_node(tree.seed_node, tree)

idx = 1
for l1_child in tree.seed_node.child_nodes():
    add_node(l1_child, tree)
    for l2_child in l1_child.child_nodes():
        add_node(l2_child, tree)
        for l3_child in l2_child.child_nodes():
            add_node(l3_child, tree)
            for l4_child in l3_child.child_nodes():
                add_node(l4_child, tree)
                for l5_child in l4_child.child_nodes():
                    l5_child.taxon = tree.taxon_namespace.get_taxon('taxon_'+str(idx))
                    idx = idx + 1
# -

tree.as_string(schema="newick")
