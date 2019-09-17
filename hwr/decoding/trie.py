# Basic Trie implementation


class TrieNode(object):
    def __init__(self):
        self.children = {}
        self.end = False
        self.counter = 1

    def get_children_nodes(self):
        return list(self.children.values())

    def get_children_chars(self):
        return list(self.children.keys())


class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def get_root(self):
        return self.root

    def insert(self, word):
        curr = self.root
        last = len(word) - 1
        for i, c in enumerate(word):
            # If not exist, create
            if c not in curr.children:
                curr.children[c] = TrieNode()
            # Move down the node
            curr = curr.children[c]
            if i == last:
                curr.end = True
        return self

    def mass_insert(self, words):
        for w in words:
            self.insert(w)

    # Return Node if exist, None if not
    def search(self, prefix):
        curr = self.root
        for c in prefix:
            if c in curr.children:
                curr = curr.children[c]
            else:
                return None
        return curr

    def is_word(self, txt):
        node = self.search(txt)
        if node:
            return node.end
        else:
            return False

    def get_char_candidates(self, txt):
        node = self.search(txt)
        ret = [*node.children] if node else []
        return ret