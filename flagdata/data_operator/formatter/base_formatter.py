import pandas as pd

class DataFormatter:
    """
    DataFormatter class defines an interface for reading and writing data in different formats.
    This class is an abstract base class, and subclasses must implement the specific read and write logic.
    """

    def read_data(self, file_path):
        """
        Reads data from a specified file path. This is an abstract method that must be implemented by subclasses.
        
        Args:
            file_path (str): The path to the data file to be read.
        
        Returns:
            DataFrame: The data read from the file, typically a pandas DataFrame.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method!")

    def write_data(self, data, file_path):
        """
        Writes data to a specified file path. This is an abstract method that must be implemented by subclasses.
        
        Args:
            data (DataFrame): The data to be written to the file, typically a pandas DataFrame.
            file_path (str): The path where the data will be written.
        
        Returns:
            None
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method!")

    def format_data(self, source_path, target_path, target_formatter):
        """
        Reads data from the specified source path and uses the target formatter to write the data to the target path.
        This method combines reading and writing functionalities, providing a simple way to transform data formats.
        
        Args:
            source_path (str): The file path from which data is to be read.
            target_path (str): The file path where the transformed data will be written.
            target_formatter (DataFormatter): The instance of the target formatter used for writing data.
        
        Returns:
            None
        """
        # Use the read_data method of this class to read source data
        data = self.read_data(source_path)
        # Use the write_data method of the passed target_formatter to write data to the target path
        target_formatter.write_data(data, target_path)
