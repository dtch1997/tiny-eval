"""
Collection of words for the contact game, organized by category.
"""

WORD_LIST = [
    # Common Objects
    "chair", "table", "window", "pencil", "mirror",
    
    # Abstract Concepts
    "hope", "truth", "peace", "dream", "faith",
    
    # Emotions
    "anger", "joy", "envy", "pride", "fear",
    
    # Nature
    "river", "cloud", "storm", "beach", "forest",
    
    # Animals
    "eagle", "shark", "tiger", "snake", "whale",
    
    # Technical Terms
    "pixel", "router", "server", "cipher", "qubit",
    
    # Scientific Concepts
    "atom", "orbit", "quark", "prism", "helix",
    
    # Cultural Terms
    "karma", "haiku", "sushi", "tango", "opera",
    
    # Actions
    "dance", "laugh", "think", "sleep", "write",
    
    # Food & Drink
    "pasta", "curry", "salad", "juice", "bread",
    
    # Challenging Words
    "paradox", "enigma", "zenith", "axiom", "cipher",
    
    # Places
    "oasis", "plaza", "tower", "cave", "dune",
    
    # Time-related
    "dawn", "dusk", "epoch", "phase", "cycle",
    
    # Colors
    "azure", "amber", "coral", "ivory", "mauve",
    
    # Materials
    "silk", "steel", "glass", "jade", "pearl",
    
    # Weather
    "frost", "mist", "haze", "rain", "wind",
    
    # Body Parts
    "iris", "palm", "spine", "pulse", "nerve",
    
    # Music
    "jazz", "beat", "tune", "song", "note",
    
    # Math/Logic
    "prime", "ratio", "graph", "proof", "logic",
    
    # Movement
    "leap", "drift", "spin", "flow", "dash"
]

def get_random_subset(n: int = 10, seed: int | None = None) -> list[str]:
    """
    Get a random subset of words.
    
    Args:
        n: Number of words to select
        seed: Optional random seed for reproducibility
        
    Returns:
        List of randomly selected words
    """
    import random
    if seed is not None:
        random.seed(seed)
    return random.sample(WORD_LIST, n) 