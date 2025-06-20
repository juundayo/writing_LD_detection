# ----------------------------------------------------------------------------#

import collections

# ----------------------------------------------------------------------------#

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
        self.is_end = False
        self.frequency = 0

    def insert(self, word: str, frequency: int=0) -> None:
        """
        Inserts a word into the trie.
        """
        current = self.root
        for letter in word:
            current = current.children[letter]

        current.is_end = True
        current.frequency = frequency

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
    
    def get_frequency(self, word: str) -> int:
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return 0
            
        return current.frequency if current.is_end else 0
    
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
    
# ----------------------------------------------------------------------------#

# Loading the Greek dictionary.
def load_greek_dictionary(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Splitting the line into word and frequency.
            parts = line.strip().split()
            if not parts:
                continue
                
            word = parts[0]
            frequency = int(parts[1]) if len(parts) > 1 else 0
            
            if word:  # Only inserting non-empty strings.
                greek_trie.insert(word, frequency)

    print('Loaded the dictionary!')

# ----------------------------------------------------------------------------#

# Creating the trie.
greek_trie = Trie()
load_greek_dictionary('/home/ml3/Desktop/Thesis/.venv/Data/filtered_dictionary.dic')

# ----------------------------------------------------------------------------#

# Testing!
print(greek_trie.search("αυτός"))
#print(greek_trie.startsWith("γει"))
