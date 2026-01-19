import re
import os

# List of markdown files to clean
files = [
    'results/thesis_visuals/README.md',
    'results/thesis_visuals/IMPROVEMENTS.md',
    'results/thesis_visuals/DATA_TRANSFORMATION_EXPLAINED.md'
]

# Comprehensive emoji pattern
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\u2600-\u26FF"          # misc symbols
    "\u2700-\u27BF"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\u2B50\u2B55"           # stars
    "\u2705\u2714"           # checkmarks
    "\u274C\u274E"           # crosses
    "\u2753-\u2757"          # question/exclamation marks
    "\u2194-\u2199"          # arrows
    "\u231A\u231B"           # watches
    "\u23E9-\u23FA"          # media buttons
    "\u25AA-\u25AB"          # squares
    "\u2934-\u2935"          # arrows
    "\U0001F910-\U0001F96B"  # supplemental emoticons
    "\U0001F980-\U0001F9E0"  # supplemental symbols
    "]", 
    flags=re.UNICODE
)

for file_path in files:
    if os.path.exists(file_path):
        print(f"Processing {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove emojis
        clean_content = emoji_pattern.sub('', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(clean_content)
        
        print(f"  ✓ Cleaned")
    else:
        print(f"  ✗ File not found: {file_path}")

print("\nAll emojis removed successfully!")
