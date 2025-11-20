from flask import Flask, render_template, request, jsonify
from newspaper import Article, Config
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass

app = Flask(__name__)

def preprocess_text(text):
    """Preprocessing steps for body content"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = text.strip()
    return text
 
def remove_stopwords(tokens, language='english'):
    """Remove stopwords"""
    stop_words = set(stopwords.words(language))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def extract_ngrams(tokens, n=3):
    """Extract n-grams (multi-word phrases)"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def extract_hot_keywords(text, top_n=20):
    """Extract hot keywords (phrases) with normalized scores"""
    cleaned_text = preprocess_text(text)
    tokens = word_tokenize(cleaned_text)
    
    # Detect if text is Arabic or English
    if any('\u0600' <= c <= '\u06FF' for c in cleaned_text):
        # Arabic text
        filtered_tokens = remove_stopwords(tokens, language='arabic')
    else:
        # English text
        filtered_tokens = remove_stopwords(tokens, language='english')
    
    filtered_tokens = [word for word in filtered_tokens if len(word) > 1]
    
    # Extract 3-grams and 4-grams (phrases)
    bigrams = extract_ngrams(filtered_tokens, n=3)
    trigrams = extract_ngrams(filtered_tokens, n=4)
    
    # Combine and count all phrases
    all_phrases = bigrams + trigrams
    phrase_freq = Counter(all_phrases)
    
    hot_keywords = phrase_freq.most_common(top_n)
    
    total_phrases = sum(phrase_freq.values()) if phrase_freq else 1
    normalized_keywords = [(phrase, freq, round(freq / total_phrases, 3)) 
                           for phrase, freq in hot_keywords]
    
    return normalized_keywords

def scrape_and_extract(url):
    """Scrape website and extract keywords"""
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        config.request_timeout = 10
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        if not article.text:
            return None, "No body content found"
        
        return article.text, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/scrape', methods=['POST'])
def scrape():
    """API endpoint to scrape and extract keywords"""
    try:
        data = request.json
        url = data.get('url', '').strip()
        top_n = int(data.get('top_n', 20))
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Scrape content
        body_text, error = scrape_and_extract(url)
        if error:
            return jsonify({'error': error}), 400
        
        # Extract keywords
        keywords = extract_hot_keywords(body_text, top_n=top_n)
        
        return jsonify({
            'success': True,
            'url': url,
            'body_length': len(body_text),
            'body_preview': body_text[:500] + '...',
            'keywords': [{'word': w[0], 'frequency': w[1], 'score': w[2]} for w in keywords]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)