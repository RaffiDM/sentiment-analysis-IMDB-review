import tensorflow as tf
import re

LABEL_KEY = "sentiment"
FEATURE_KEY = "review"

def transformed_name(key):
    """Mengganti nama fitur yang ditransformasi"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Melakukan preprocessing pada fitur input menjadi fitur yang ditransformasi
    
    Args:
        inputs: map dari feature keys ke raw features.
    
    Return:
        outputs: map dari feature keys ke transformed features.    
    """
    
    outputs = {}
    # Mengganti tag HTML <br>, <...> dengan spasi
    cleaned_reviews = tf.strings.regex_replace(inputs[FEATURE_KEY], r'<br\s*/?>|<.*?>', ' ')

    # Mengganti karakter HTML seperti &quot;, &amp;, &lt;, &gt; dengan spasi
    cleaned_reviews = tf.strings.regex_replace(cleaned_reviews, r'&quot;|&amp;|&lt;|&gt;', ' ')

    # remove punctuation
    cleaned_reviews = tf.strings.regex_replace(cleaned_reviews, r'[^\w\s]', ' ')

    # normalisasi spasi
    cleaned_reviews = tf.strings.regex_replace(cleaned_reviews, r'\s+', ' ') 

    # Menghapus spasi di awal dan akhir kalimat
    cleaned_reviews = tf.strings.strip(cleaned_reviews)
    
    # Mengubah teks menjadi huruf kecil
    cleaned_reviews = tf.strings.lower(cleaned_reviews) 
    
    # Store the transformed reviews
    outputs[transformed_name(FEATURE_KEY)] = cleaned_reviews
    
    # Convert label (1=positive, 0=negative) to int64
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
