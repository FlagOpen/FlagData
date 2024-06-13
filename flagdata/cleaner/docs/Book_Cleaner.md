## Book_Cleaner
### Description

`Book_Cleaner` is a module designed for processing e-books. Its main function is to extract text content from e-book files (formats like EPUB, MOBI, AZW) and remove unwanted parts such as the table of contents, advertisements and image annotations.

### Environment Setup

`Book_Cleaner` utilizes the `Calibre ` library ([[https://calibre-ebook.com/](https://calibre-ebook.com/)])  to parse e-books.

You can install `Calibre ` with the following command:


`sudo -v && wget -nv -O- https://download.calibre-ebook.com/linux-installer.sh | sudo sh /dev/stdin`

### Usage
`Book_Cleaner` is initialized with three parameters:


通过调用 `Book_Cleaner`的`clean`函数可以实现对电子书的处理，以下是一个简单的使用示例：

1. `config_path`: Path to a JSON file containing a series of keywords that guide the text cleaning process. Users can edit this file to add custom keywords to suit specific cleaning needs.

2. `workspace`: A directory path used to store temporary files generated during the e-book processing.

   (**Note**: When `Book_Cleaner` is initialized, it will create a folder named `book_temp_dir1` in `workspace`. Each time the clean function is called, all files inside `book_temp_dir1` will be deleted. If a folder named `book_temp_dir1` already exists at initialization, it will attempt to create `book_temp_dir2`, and so forth.)

3. `target_path`: Path to a JSON file where the processed content will be saved.

The `clean` function of  `Book_Cleaner` can be used to process e-books. Here is a simple usage example:


```python
# Configuration file path
config_path = 'path/to/config.json'
# Temporary file path
workspace = 'path/to/workspace'
# Target path
target_path = 'path/to/target'

# Initialize
cleaner = Book_Cleaner(config_path, workspace, target_path)

# Perform cleaning
cleaner.process_books()
```

### Demonstration
#### 1. Remove cover, copyright page
***
>~~*First published in 2015*~~
>
>~~*Allen & Unwin*~~
>
>~~*83 Alexander Street*~~
>
>~~*Crows Nest NSW 2065*~~
>
>~~*Australia*~~
>
>~~*Phone: (61 2) 8425 0100*~~
>
>~~*Cataloguing-in-Publication details are available from the National Library of Australia*~~
>
>~~*ISBN 978 1 74331 9208*~~
>
>~~*eISBN 978 1 74343 637 0*~~
>
>~~*Internal design by Christabella Designs*~~
>
>~~*Typeset by Post Pre-press Group, Australia*~~
>
>~~*Extracts from The Kite Runner by Khaled Hosseini © Bloomsbury Publishing UK*~~
>
> …
>
>*Chapter 1*
>
>*She was standing at the end of the hall in the late afternoon light, her back arched against the wall, between two landscape paintings. Closing her eyes, she ran her hands up and down the red dress clinging to her curves. The dress had been designed by a genius. Its neckline was low enough to be provocative, but the hemline was below knee-length—too long for Leo, the husband of our hostess, to …*
***
#### 2. Removing Directories
***
>~~*Contents*~~
>
>~~*Cover*~~
>
>~~*About the Book*~~
>
>~~*About the Author*~~
>
>~~*Title Page*~~
>
>~~*Dedication*~~
>
>~~*Prologue*~~
>
>~~*Chapter One Innocent Days*~~
>
>~~*Chapter Two Culture Shock*~~
>
>~~*Chapter Three Daddy*~~
>
>…
>
>*About the Book*
>
>*What do they find attractive about me? An underage girl who just lies there sobbing, looking up at them... as they come to me one by one.*
>*This is the shocking true story of how a young girl from Rochdale came to be Girl A – the key witness in the trial of Britain’s most notorious child sex ring. …*
***
#### 3. Remove image annotations
***
>*No disease has ever been so instantly recognized or so widely known and feared. …*
>
>~~*Figure 1. Smallpox Deities. Sopona (left) was the smallpox god among the Yorubas of western Africa. Sitala Mata (right), the Hindu goddess of smallpox, shown astride a donkey, was widely worshipped in temples throughout the Indian countryside.*~~
>
>*Dr. Nick Ward, one of my senior staff, accompanied me on the ward rounds. A veteran of medical service in Africa, he had cared for patients with the worst of tropical diseases. As we left the hospital, he placed his hands on the railing of a balcony, leaned over as he looked at the ground and said, “I don’t think I can ever again walk through a ward like that. It is unimaginable.”*
***

#### 4. Remove advertisements
***
>~~*Medieval Series*~~
>
>~~*Legendary Bastards of the Crown Series*~~
>
>~~*Legendary Bastards of the Crown Series*~~
>
>~~*Seasons of Fortitude Series*~~
>
>~~*Legacy of the Blade Series*~~
>
>*…*
>
>~~*And More!*~~
>
>~~*Please visit http:// xxx.com !*~~
>
>*…*
>
>*A small hum of excitement rose inside her. She’d been traveling since the morning, and this trip was twice the distance of the longest journey she’d ever taken alone. But she’d done it. Exhausted but quietly exhilarated, she turned the radio up loud. This was a time for celebration. She clenched her hand in the air in a solitary fist-bump.*
>*A wooden sign on the left-hand side of the road read: BLEATH Population 3,667*
>
>~~*Good book recommendation: http:// xxx.com !*~~
>
>*A small town was going to be a culture shock after New York City—the only place she’d ever known. She’d lived in a three-bedroom apartment with her parents and brother since she was three. And now she was heading off by herself for an entire month, to conduct research for her final year thesis at college.*
***


