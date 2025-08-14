# ----------------------------------------------------------------------------#

import collections
import time

# ----------------------------------------------------------------------------#

TESTING = True 

# ----------------------------------------------------------------------------#

class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_end = False

# ----------------------------------------------------------------------------#

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
    
    def suggest_correction(self, word: str, max_distance: int = 2) -> str:
        """
        Returns the most likely correct word from the trie for a given misspelled word.
        Prioritizes edit distance first, then uses frequency only as a tiebreaker.
        """
        suggestions = []
        
        def dfs(node, current_word, remaining_edits, original_word, position=0):
            if remaining_edits < 0:
                return
            
            if position >= len(original_word):
                if node.is_end:
                    # Calculate actual edit distance (not just remaining edits).
                    actual_distance = self.calculate_edit_distance(current_word, original_word)
                    suggestions.append((current_word, node.frequency, actual_distance))
                # Try adding remaining letters from the trie.
                for char, child in node.children.items():
                    dfs(child, current_word + char, remaining_edits - 1, original_word, position)
                return
            
            current_char = original_word[position]
            
            # Exact match - no edit needed.
            if current_char in node.children:
                dfs(node.children[current_char], current_word + current_char, remaining_edits, original_word, position + 1)
            
            if remaining_edits > 0:
                # Insertion (skipping the current char).
                dfs(node, current_word, remaining_edits - 1, original_word, position + 1)
                
                # Deletion (adding a new char).
                for char, child in node.children.items():
                    dfs(child, current_word + char, remaining_edits - 1, original_word, position)
                
                # Substitution.
                for char, child in node.children.items():
                    if char != current_char:
                        dfs(child, current_word + char, remaining_edits - 1, original_word, position + 1)
        
        # Starting the search from the root.
        dfs(self.root, "", max_distance, word)
        
        if not suggestions:
            return word  # Returning the original if no suggestions are found.
        
        # Sort suggestions by: 1. smallest edit distance, 2. highest frequency.
        suggestions.sort(key=lambda x: (x[2], -x[1]))
        
        # Returning the best suggestion.
        return suggestions[0][0]

    def calculate_edit_distance(self, word1, word2):
        """
        Helper function to calculate actual 
        Levenshtein distance between two words.
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], # Deletion.
                                    dp[i][j-1],    # Insertion.
                                    dp[i-1][j-1])  # Substitution.
        return dp[m][n]
    
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
if TESTING:
    '''
    start = time.time()
    print(greek_trie.search("αυτός"))
    end = time.time()
    legth = end - start
    print(f"Trie search took {legth:.6f} seconds.")

    print(greek_trie.startsWith("εκείν"))
    '''

    print("Testing word suggestions:")
    test_words = ["ανθοπολίο", "σκιλί", "συνεφιασμένο", "σχολίο", "βασιλισσα"]
    
    for word in test_words:
        if not greek_trie.search(word):
            suggestion = greek_trie.suggest_correction(word)
            print(f"Did you mean '{suggestion}' instead of '{word}'?")
        else:
            print(f"'{word}' is correctly spelled!")
