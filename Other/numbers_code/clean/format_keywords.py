import ast

def format_data(row):
    try:
        # Using ast.literal_eval to safely evaluate the string as a Python literal
        formatted_row = ast.literal_eval(row)
        return formatted_row
    except (SyntaxError, ValueError) as e:
        # Handle the case where the string is not a valid Python literal
        print(f"Error formatting row: {row}. Error: {e}")
        return None

# Example usage
original_data = "['Gothic Horror', ""Frankenstein's Monster"", 'Monster']"
formatted_data = format_data(original_data)

if formatted_data is not None:
    print(formatted_data)
