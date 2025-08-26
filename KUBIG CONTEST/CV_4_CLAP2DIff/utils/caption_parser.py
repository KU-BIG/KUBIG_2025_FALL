"""
Caption Parser for AudioCaps Dataset
Parses complex audio captions to extract hierarchical information
"""

import re
from typing import Dict, List, Tuple, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag


class AudioCaptionParser:
    """
    Parses AudioCaps captions to extract hierarchical audio information:
    - Primary events (foreground)
    - Secondary events (background)
    - Environmental context (ambience)
    """
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Keywords for identifying relationships
        self.temporal_keywords = ['while', 'as', 'during', 'when']
        self.additive_keywords = ['and', 'with', 'along with', 'as well as']
        self.environmental_keywords = ['in', 'at', 'inside', 'outside', 'near', 'by']
        self.background_keywords = ['in the background', 'in the distance', 'faintly', 'softly']
        
        # Sound category mappings
        self.sound_categories = {
            'human': ['talk', 'speak', 'voice', 'laugh', 'cry', 'shout', 'sing', 'whisper'],
            'animal': ['bark', 'meow', 'chirp', 'roar', 'howl', 'moo', 'neigh'],
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'engine', 'horn', 'brake'],
            'nature': ['wind', 'rain', 'thunder', 'water', 'wave', 'storm', 'leaves'],
            'music': ['music', 'instrument', 'piano', 'guitar', 'drum', 'violin'],
            'mechanical': ['machine', 'motor', 'fan', 'drill', 'saw', 'pump'],
            'impact': ['bang', 'crash', 'hit', 'knock', 'slam', 'break', 'shatter']
        }
    
    def parse_caption(self, caption: str) -> Dict[str, any]:
        """
        Parse an AudioCaps caption into hierarchical components
        
        Args:
            caption: AudioCaps caption string
            
        Returns:
            Dictionary with parsed components:
            - primary: Main sound events (foreground)
            - secondary: Background sounds
            - context: Environmental/ambience information
            - relationships: How sounds relate to each other
        """
        caption = caption.lower().strip()
        
        # Extract hierarchical components
        primary, secondary, context = self._extract_hierarchy(caption)
        
        # Identify sound categories
        categories = self._identify_categories(caption)
        
        # Determine relationships
        relationships = self._analyze_relationships(caption)
        
        # Extract action words (verbs)
        actions = self._extract_actions(caption)
        
        return {
            'original': caption,
            'primary': primary,
            'secondary': secondary,
            'context': context,
            'categories': categories,
            'relationships': relationships,
            'actions': actions,
            'complexity': self._estimate_complexity(caption)
        }
    
    def _extract_hierarchy(self, caption: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract primary, secondary, and context information
        """
        primary = []
        secondary = []
        context = []
        
        # Check for background indicators
        if any(keyword in caption for keyword in self.background_keywords):
            # Split by background keywords
            for keyword in self.background_keywords:
                if keyword in caption:
                    parts = caption.split(keyword)
                    if len(parts) > 1:
                        primary.append(parts[0].strip())
                        secondary.append(parts[1].strip())
        
        # Check for temporal relationships (simultaneous events)
        for keyword in self.temporal_keywords:
            if keyword in caption:
                parts = caption.split(keyword)
                if len(parts) > 1:
                    primary.append(parts[0].strip())
                    secondary.append(parts[1].strip())
        
        # Check for environmental context
        for keyword in self.environmental_keywords:
            if f' {keyword} ' in caption:
                # Extract the environmental description
                pattern = f'{keyword}\\s+([a-z\\s]+?)(?:,|\\.|$|and|while)'
                matches = re.findall(pattern, caption)
                context.extend(matches)
        
        # If no specific hierarchy found, treat entire caption as primary
        if not primary and not secondary:
            # Split by 'and' for multiple events
            if ' and ' in caption:
                events = caption.split(' and ')
                primary = [events[0]] if events else []
                secondary = events[1:] if len(events) > 1 else []
            else:
                primary = [caption]
        
        # Clean up extracted components
        primary = [self._clean_text(p) for p in primary if p]
        secondary = [self._clean_text(s) for s in secondary if s]
        context = [self._clean_text(c) for c in context if c]
        
        return primary, secondary, context
    
    def _identify_categories(self, caption: str) -> List[str]:
        """
        Identify sound categories present in the caption
        """
        categories = []
        
        for category, keywords in self.sound_categories.items():
            if any(keyword in caption for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def _analyze_relationships(self, caption: str) -> str:
        """
        Determine the relationship type between sounds
        """
        # Check for temporal relationships
        if any(keyword in caption for keyword in self.temporal_keywords):
            return "simultaneous"
        
        # Check for additive relationships
        if any(keyword in caption for keyword in self.additive_keywords):
            return "additive"
        
        # Check for spatial relationships
        if any(keyword in caption for keyword in self.environmental_keywords):
            return "spatial"
        
        # Default to single event
        return "single"
    
    def _extract_actions(self, caption: str) -> List[str]:
        """
        Extract action words (verbs) from the caption
        """
        tokens = word_tokenize(caption)
        pos_tags = pos_tag(tokens)
        
        # Extract verbs
        verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
        
        return verbs
    
    def _estimate_complexity(self, caption: str) -> str:
        """
        Estimate the complexity of the audio scene
        """
        # Count number of distinct events
        event_count = len(caption.split(' and ')) + len(caption.split(' while '))
        
        if event_count >= 3:
            return "complex"
        elif event_count == 2:
            return "moderate"
        else:
            return "simple"
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text
        """
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Remove leading/trailing punctuation
        text = text.strip('.,;:')
        
        # Remove articles at the beginning
        articles = ['a ', 'an ', 'the ']
        for article in articles:
            if text.startswith(article):
                text = text[len(article):]
        
        return text.strip()
    
    def get_hierarchy_labels(self, parsed_caption: Dict) -> Dict[str, str]:
        """
        Generate labels for hierarchical decomposition training
        
        Args:
            parsed_caption: Parsed caption dictionary
            
        Returns:
            Labels for foreground, background, and ambience
        """
        labels = {
            'foreground': ' '.join(parsed_caption['primary'][:1]),  # Main event
            'background': ' '.join(parsed_caption['secondary'][:1]) if parsed_caption['secondary'] else '',
            'ambience': ' '.join(parsed_caption['context']) if parsed_caption['context'] else ''
        }
        
        # Fill empty slots with descriptive defaults based on categories
        if not labels['background'] and parsed_caption['categories']:
            labels['background'] = f"{parsed_caption['categories'][0]} sounds"
        
        if not labels['ambience']:
            if parsed_caption['complexity'] == 'complex':
                labels['ambience'] = "busy environment"
            elif parsed_caption['complexity'] == 'simple':
                labels['ambience'] = "quiet setting"
            else:
                labels['ambience'] = "ambient sounds"
        
        return labels


def test_parser():
    """
    Test the caption parser with AudioCaps examples
    """
    parser = AudioCaptionParser()
    
    # Test captions from AudioCaps
    test_captions = [
        "A woman speaks while a dog barks in the background",
        "Cars passing by as rain falls",
        "Music playing with people talking and laughing",
        "A man giving a speech in a crowded room",
        "Thunder rumbling in the distance while wind blows",
        "A cat meows and a door opens",
        "Children playing at a playground with birds chirping"
    ]
    
    print("Testing AudioCaps Caption Parser\n" + "="*50)
    
    for caption in test_captions:
        print(f"\nOriginal: {caption}")
        parsed = parser.parse_caption(caption)
        
        print(f"Primary: {parsed['primary']}")
        print(f"Secondary: {parsed['secondary']}")
        print(f"Context: {parsed['context']}")
        print(f"Categories: {parsed['categories']}")
        print(f"Relationship: {parsed['relationships']}")
        print(f"Complexity: {parsed['complexity']}")
        
        # Get hierarchy labels
        labels = parser.get_hierarchy_labels(parsed)
        print(f"Hierarchy Labels:")
        print(f"  Foreground: {labels['foreground']}")
        print(f"  Background: {labels['background']}")
        print(f"  Ambience: {labels['ambience']}")
        print("-" * 30)


if __name__ == "__main__":
    test_parser()