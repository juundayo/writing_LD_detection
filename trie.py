import collections

class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_end = False

class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_end = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_end
    
    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie 
        that starts with the given prefix.
        """
        current = self.root 
        
        for letter in prefix:
            current = current.children.get(letter)
            if not current:
                return False
        
        return True
