import csv
import io

class CSVFormatter:
    """Formatter to convert data into CSV format."""
    
    def format(self, samples):
        """Convert samples to CSV format string."""
        if not samples:
            return ""
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=samples[0].keys())
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)
        
        return output.getvalue()

# 使用示例
csv_formatter = CSVFormatter()
samples = [
    {'name': 'John', 'age': 30, 'city': 'New York'},
    {'name': 'Anna', 'age': 22, 'city': 'London'},
    {'name': 'Mike', 'age': 32, 'city': 'San Francisco'}
]
csv_content = csv_formatter.format(samples)
print("CSV Content:\n", csv_content)
