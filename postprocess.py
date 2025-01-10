import spacy
from typing import List, Dict
import re
import torch
from transformers import pipeline
from scipy.spatial.distance import cosine
import numpy as np

spacy.prefer_gpu()


class TranscriptProcessor:
    def __init__(self, language: str = 'en', max_length: int = 512, device: int = 0):
        """
        Initialize the transcript processor.
        
        Args:
            language (str): Language code ('en' for English, 'da' for Danish, etc.)
        """

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load appropriate spaCy model based on language
        try:
            if language == 'en':
                self.nlp = spacy.load('en_core_web_trf')
            else:
                self.nlp = spacy.load('xx_sent_ud_sm')
            # Add more languages as needed
        except OSError:
            raise ValueError(f"Please install the required language model: python -m spacy download {language}_core_news_sm")
        
        # Enable GPU for spaCy if available
        if self.device == "cuda":
            spacy.require_gpu()
            self.nlp.to(self.device)
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_encoder = pipeline("feature-extraction",
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",   
            max_length=max_length,
            truncation=True,
            device=0 if self.device == "cuda" else -1
        )
        self.max_length = max_length

        # Batch size for processing multiple sentences
        self.batch_size = 32 if self.device == "cuda" else 8

    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """
        Get embedding for a single sentence and ensure it's 1-D.
        """
        # Get the embedding and convert to numpy
        embedding = np.array(self.sentence_encoder(sentence))
        # Squeeze out any singleton dimensions and take mean over sequence length
        return np.squeeze(embedding).mean(axis=0)

    def get_batch_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a batch of sentences, ensuring each is 1-D.
        """
        if not sentences:
            return []
            
        # Get embeddings for the batch
        batch_embeddings = self.sentence_encoder(sentences)
        # Convert each embedding to 1-D numpy array
        return [np.squeeze(np.array(emb)).mean(axis=0) for emb in batch_embeddings]


    def clean_text(self, text: str) -> str:
        """
        Clean the raw transcript text.
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Fix common issues with numbers and units
        text = re.sub(r'(\d+) ?([a-zA-Z]+)', r'\1 \2', text)
        # Add spaces after punctuation if missing
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        # Remove newlines
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())
        return text.strip()

    def add_punctuation(self, text: str) -> str:
        """
        Add missing punctuation using spaCy's sentence detection.
        """
        doc = self.nlp(text)
        sentences = []
        for sent in doc.sents:
            sentence = sent.text.strip()
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            sentences.append(sentence)
        return ' '.join(sentences)

    def create_paragraphs(self, text: str, similarity_threshold: float = 0.8) -> List[str]:
        """
        Split text into paragraphs with GPU-optimized batch processing.
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        if not sentences:
            return [text]
        
        # Calculate embeddings for all sentences in batches
        all_embeddings = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            batch_embeddings = self.get_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Verify embeddings are 1-D
        for i, emb in enumerate(all_embeddings):
            if len(emb.shape) != 1:
                print(f"Warning: embedding {i} has shape {emb.shape}")
                all_embeddings[i] = np.squeeze(emb).mean(axis=0) if len(emb.shape) > 1 else emb
        
        paragraphs = []
        current_paragraph = []
        
        for i in range(len(sentences)):
            if i == 0:
                current_paragraph.append(sentences[i])
                continue
            
            try:
                similarity = 1 - cosine(all_embeddings[i], all_embeddings[i-1])
            except ValueError as e:
                print(f"Error processing embeddings at index {i}")
                print(f"Shape of current embedding: {all_embeddings[i].shape}")
                print(f"Shape of previous embedding: {all_embeddings[i-1].shape}")
                raise e
            
            if similarity >= similarity_threshold:
                current_paragraph.append(sentences[i])
            else:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sentences[i]]
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return paragraphs
    
    def identify_sections(self, paragraphs: List[str]) -> Dict[str, List[str]]:
        """
        Group paragraphs into sections based on content similarity.
        """
        sections = {}
        current_section = []
        section_count = 1
        
        for i, para in enumerate(paragraphs):
           # Start a new section if it's the first paragraph or meets certain criteria
            if (i == 0 or 
                len(current_section) >= 5 or 
                (i > 0 and len(self.nlp(para)[0].text) > 50)):  # New section for longer opening sentences
                
                if current_section:
                    sections[f"Section {section_count}"] = current_section
                    section_count += 1
                    current_section = []
            
            current_section.append(para)
        
        if current_section:
            sections[f"Section {section_count}"] = current_section
        
        return sections
    
    def fix_capitalization(self, text: str) -> str:
        """
        Fix capitalization in text:
        - Capitalize sentence starts
        - Preserve proper nouns
        """
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            # Get sentence text and strip whitespace
            sent_text = sent.text.strip()
            
            # Capitalize first letter of sentence
            if sent_text:
                sent_text = sent_text[0].upper() + sent_text[1:]
                
            sentences.append(sent_text)
        
        return ' '.join(sentences)
    

    def process_transcript(self, text: str) -> str:
        """
        Process the entire transcript.
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Fix capitalization
        capitalized_text = self.fix_capitalization(cleaned_text)

        # Add punctuation
        punctuated_text = self.add_punctuation(capitalized_text)
        
        # Create paragraphs
        paragraphs = self.create_paragraphs(punctuated_text)
        
        # Identify sections
        sections = self.identify_sections(paragraphs)
        
        # Format the final output
        output = []
        for section_name, section_paragraphs in sections.items():
            #output.append(f"\n{section_name}\n{'='*len(section_name)}\n")
            output.append(f"\n")
            output.extend([f"{para}\n" for para in section_paragraphs])
        
        return ''.join(output)
    


# Example usage
if __name__ == "__main__":
    # Enable GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Read transcript text from txt file
    with open('transcripts/transcribe-in.txt', 'r') as file:
        sample_transcript = file.read()
    
    # Initialize processor for English
    processor = TranscriptProcessor(language='en')
    print(f"Using device: {processor.device}")

    # Process the transcript
    processed_text = processor.process_transcript(sample_transcript)
    print(processed_text)

    # Save the processed text to a file
    with open('transcripts/processed-out.txt', 'w') as file:
        file.write(processed_text)