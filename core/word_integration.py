import win32com.client as win32
from grammar_checker import VietnameseGrammarChecker

def process_document(input_file, output_file, checker):
    """Process a .docx file: read text, check grammar, apply corrections, save."""
    # Initialize Word application
    word = win32.gencache.EnsureDispatch('Word.Application')
    word.Visible = False  # Run in background

    try:
        # Open the document
        doc = word.Documents.Open(input_file)

        # Read the full text
        full_text = doc.Content.Text

        # Detect errors
        errors = checker.detect_errors(full_text)

        # For each error, suggest correction (placeholder)
        for error_sentence, prob in errors:
            # In a real implementation, replace in document
            # For simplicity, print errors
            print(f"Potential error: {error_sentence} (prob: {prob})")

        # Save the document (no changes yet)
        doc.SaveAs(output_file)
        doc.Close()

    finally:
        word.Quit()

    return errors
